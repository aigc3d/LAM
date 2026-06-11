import json
import time
import warnings
import numpy as np
from typing import List, Optional,Tuple
from scipy.signal import savgol_filter


ARKitLeftRightPair = [
        ("jawLeft", "jawRight"),
        ("mouthLeft", "mouthRight"),
        ("mouthSmileLeft", "mouthSmileRight"),
        ("mouthFrownLeft", "mouthFrownRight"),
        ("mouthDimpleLeft", "mouthDimpleRight"),
        ("mouthStretchLeft", "mouthStretchRight"),
        ("mouthPressLeft", "mouthPressRight"),
        ("mouthLowerDownLeft", "mouthLowerDownRight"),
        ("mouthUpperUpLeft", "mouthUpperUpRight"),
        ("cheekSquintLeft", "cheekSquintRight"),
        ("noseSneerLeft", "noseSneerRight"),
        ("browDownLeft", "browDownRight"),
        ("browOuterUpLeft", "browOuterUpRight"),
        ("eyeBlinkLeft","eyeBlinkRight"),
        ("eyeLookDownLeft","eyeLookDownRight"),
        ("eyeLookInLeft", "eyeLookInRight"),
        ("eyeLookOutLeft","eyeLookOutRight"),
        ("eyeLookUpLeft","eyeLookUpRight"),
        ("eyeSquintLeft","eyeSquintRight"),
        ("eyeWideLeft","eyeWideRight")
    ]

ARKitBlendShape =[
   "browDownLeft",
   "browDownRight",
   "browInnerUp",
   "browOuterUpLeft",
   "browOuterUpRight",
   "cheekPuff",
   "cheekSquintLeft",
   "cheekSquintRight",
   "eyeBlinkLeft",
   "eyeBlinkRight",
   "eyeLookDownLeft",
   "eyeLookDownRight",
   "eyeLookInLeft",
   "eyeLookInRight",
   "eyeLookOutLeft",
   "eyeLookOutRight",
   "eyeLookUpLeft",
   "eyeLookUpRight",
   "eyeSquintLeft",
   "eyeSquintRight",
   "eyeWideLeft",
   "eyeWideRight",
   "jawForward",
   "jawLeft",
   "jawOpen",
   "jawRight",
   "mouthClose",
   "mouthDimpleLeft",
   "mouthDimpleRight",
   "mouthFrownLeft",
   "mouthFrownRight",
   "mouthFunnel",
   "mouthLeft",
   "mouthLowerDownLeft",
   "mouthLowerDownRight",
   "mouthPressLeft",
   "mouthPressRight",
   "mouthPucker",
   "mouthRight",
   "mouthRollLower",
   "mouthRollUpper",
   "mouthShrugLower",
   "mouthShrugUpper",
   "mouthSmileLeft",
   "mouthSmileRight",
   "mouthStretchLeft",
   "mouthStretchRight",
   "mouthUpperUpLeft",
   "mouthUpperUpRight",
   "noseSneerLeft",
   "noseSneerRight",
   "tongueOut"
]

MOUTH_BLENDSHAPES = [ "mouthDimpleLeft",
                    "mouthDimpleRight",
                    "mouthFrownLeft",
                    "mouthFrownRight",
                    "mouthFunnel",
                    "mouthLeft",
                    "mouthLowerDownLeft",
                    "mouthLowerDownRight",
                    "mouthPressLeft",
                    "mouthPressRight",
                    "mouthPucker",
                    "mouthRight",
                    "mouthRollLower",
                    "mouthRollUpper",
                    "mouthShrugLower",
                    "mouthShrugUpper",
                    "mouthSmileLeft",
                    "mouthSmileRight",
                    "mouthStretchLeft",
                    "mouthStretchRight",
                    "mouthUpperUpLeft",
                    "mouthUpperUpRight",
                    "jawForward",
                    "jawLeft",
                    "jawOpen",
                    "jawRight",
                    "noseSneerLeft",
                    "noseSneerRight",
                    "cheekPuff",
                ]

DEFAULT_CONTEXT ={
    'is_initial_input': True,
    'previous_audio': None,
    'previous_expression': None,
    'previous_volume': None,
    'previous_headpose': None,
}

RETURN_CODE = {
    "SUCCESS": 0,
    "AUDIO_LENGTH_ERROR": 1,
    "CHECKPOINT_PATH_ERROR":2,
    "MODEL_INFERENCE_ERROR":3,
}

DEFAULT_CONTEXTRETURN = {
    "code": RETURN_CODE['SUCCESS'],
    "expression": None,
    "headpose": None,
}

BLINK_PATTERNS = [
    np.array([0.365, 0.950, 0.956, 0.917, 0.367, 0.119, 0.025]),
    np.array([0.235, 0.910, 0.945, 0.778, 0.191, 0.235, 0.089]),
    np.array([0.870, 0.950, 0.949, 0.696, 0.191, 0.073, 0.007]),
    np.array([0.000, 0.557, 0.953, 0.942, 0.426, 0.148, 0.018])
]

# Postprocess
def symmetrize_blendshapes(
        bs_params: np.ndarray,
        mode: str = "average",
        symmetric_pairs: list = ARKitLeftRightPair
) -> np.ndarray:
    """
    Apply symmetrization to ARKit blendshape parameters (batched version)

    Args:
        bs_params: numpy array of shape (N, 52), batch of ARKit parameters
        mode: symmetrization mode ["average", "max", "min", "left_dominant", "right_dominant"]
        symmetric_pairs: list of left-right parameter pairs

    Returns:
        Symmetrized parameters with same shape (N, 52)
    """

    name_to_idx = {name: i for i, name in enumerate(ARKitBlendShape)}

    # Input validation
    if bs_params.ndim != 2 or bs_params.shape[1] != 52:
        raise ValueError("Input must be of shape (N, 52)")

    symmetric_bs = bs_params.copy()  # Shape (N, 52)

    # Precompute valid index pairs
    valid_pairs = []
    for left, right in symmetric_pairs:
        left_idx = name_to_idx.get(left)
        right_idx = name_to_idx.get(right)
        if None not in (left_idx, right_idx):
            valid_pairs.append((left_idx, right_idx))

    # Vectorized processing
    for l_idx, r_idx in valid_pairs:
        left_col = symmetric_bs[:, l_idx]
        right_col = symmetric_bs[:, r_idx]

        if mode == "average":
            new_vals = (left_col + right_col) / 2
        elif mode == "max":
            new_vals = np.maximum(left_col, right_col)
        elif mode == "min":
            new_vals = np.minimum(left_col, right_col)
        elif mode == "left_dominant":
            new_vals = left_col
        elif mode == "right_dominant":
            new_vals = right_col
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Update both columns simultaneously
        symmetric_bs[:, l_idx] = new_vals
        symmetric_bs[:, r_idx] = new_vals

    return symmetric_bs


def apply_random_eye_blinks(
        input: np.ndarray,
        blink_scale: tuple = (0.8, 1.0),
        blink_interval: tuple = (60, 120),
        blink_duration: int = 7
) -> np.ndarray:
    """
    Apply randomized eye blinks to blendshape parameters

    Args:
        output: Input array of shape (N, 52) containing blendshape parameters
        blink_scale: Tuple (min, max) for random blink intensity scaling
        blink_interval: Tuple (min, max) for random blink spacing in frames
        blink_duration: Number of frames for blink animation (fixed)

    Returns:
        None (modifies output array in-place)
    """
    # Define eye blink patterns (normalized 0-1)

    # Initialize parameters
    n_frames = input.shape[0]
    input[:,8:10] = np.zeros((n_frames,2))
    current_frame = 0

    # Main blink application loop
    while current_frame < n_frames - blink_duration:
        # Randomize blink parameters
        scale = np.random.uniform(*blink_scale)
        pattern = BLINK_PATTERNS[np.random.randint(0, 4)]

        # Apply blink animation
        blink_values = pattern * scale
        input[current_frame:current_frame + blink_duration, 8] = blink_values
        input[current_frame:current_frame + blink_duration, 9] = blink_values

        # Advance to next blink position
        current_frame += blink_duration + np.random.randint(*blink_interval)

    return input


def apply_random_eye_blinks_context(
        animation_params: np.ndarray,
        processed_frames: int = 0,
        intensity_range: tuple = (0.8, 1.0)
) -> np.ndarray:
    """Applies random eye blink patterns to facial animation parameters.

    Args:
        animation_params: Input facial animation parameters array with shape [num_frames, num_features].
                          Columns 8 and 9 typically represent left/right eye blink parameters.
        processed_frames: Number of already processed frames that shouldn't be modified
        intensity_range: Tuple defining (min, max) scaling for blink intensity

    Returns:
        Modified animation parameters array with random eye blinks added to unprocessed frames
    """
    remaining_frames = animation_params.shape[0] - processed_frames

    # Only apply blinks if there's enough remaining frames (blink pattern requires 7 frames)
    if remaining_frames <= 7:
        return animation_params

    # Configure blink timing parameters
    min_blink_interval = 40  # Minimum frames between blinks
    max_blink_interval = 100  # Maximum frames between blinks

    # Find last blink in previously processed frames (column 8 > 0.5 indicates blink)
    previous_blink_indices = np.where(animation_params[:processed_frames, 8] > 0.5)[0]
    last_processed_blink = previous_blink_indices[-1] - 7 if previous_blink_indices.size > 0 else processed_frames

    # Calculate first new blink position
    blink_interval = np.random.randint(min_blink_interval, max_blink_interval)
    first_blink_start = max(0, blink_interval - last_processed_blink)

    # Apply first blink if there's enough space
    if first_blink_start <= (remaining_frames - 7):
        # Randomly select blink pattern and intensity
        blink_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
        intensity = np.random.uniform(*intensity_range)

        # Calculate blink frame range
        blink_start = processed_frames + first_blink_start
        blink_end = blink_start + 7

        # Apply pattern to both eyes
        animation_params[blink_start:blink_end, 8] = blink_pattern * intensity
        animation_params[blink_start:blink_end, 9] = blink_pattern * intensity

        # Check space for additional blink
        remaining_after_blink = animation_params.shape[0] - blink_end
        if remaining_after_blink > min_blink_interval:
            # Calculate second blink position
            second_intensity = np.random.uniform(*intensity_range)
            second_interval = np.random.randint(min_blink_interval, max_blink_interval)

            if (remaining_after_blink - 7) > second_interval:
                second_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
                second_blink_start = blink_end + second_interval
                second_blink_end = second_blink_start + 7

                # Apply second blink
                animation_params[second_blink_start:second_blink_end, 8] = second_pattern * second_intensity
                animation_params[second_blink_start:second_blink_end, 9] = second_pattern * second_intensity

    return animation_params


def export_blendshape_animation(
        blendshape_weights: np.ndarray,
        output_path: str,
        blendshape_names: List[str],
        fps: float,
        rotation_data: Optional[np.ndarray] = None
) -> None:
    """
    Export blendshape animation data to JSON format compatible with ARKit.

    Args:
        blendshape_weights: 2D numpy array of shape (N, 52) containing animation frames
        output_path: Full path for output JSON file (including .json extension)
        blendshape_names: Ordered list of 52 ARKit-standard blendshape names
        fps: Frame rate for timing calculations (frames per second)
        rotation_data: Optional 3D rotation data array of shape (N, 3)

    Raises:
        ValueError: If input dimensions are incompatible
        IOError: If file writing fails
    """
    # Validate input dimensions
    if blendshape_weights.shape[1] != 52:
        raise ValueError(f"Expected 52 blendshapes, got {blendshape_weights.shape[1]}")
    if len(blendshape_names) != 52:
        raise ValueError(f"Requires 52 blendshape names, got {len(blendshape_names)}")
    if rotation_data is not None and len(rotation_data) != len(blendshape_weights):
        raise ValueError("Rotation data length must match animation frames")

    # Build animation data structure
    animation_data = {
        "names":blendshape_names,
        "metadata": {
            "fps": fps,
            "frame_count": len(blendshape_weights),
            "blendshape_names": blendshape_names
        },
        "frames": []
    }

    # Convert numpy array to serializable format
    for frame_idx in range(blendshape_weights.shape[0]):
        frame_data = {
            "weights": blendshape_weights[frame_idx].tolist(),
            "time": frame_idx / fps,
            "rotation": rotation_data[frame_idx].tolist() if rotation_data else []
        }
        animation_data["frames"].append(frame_data)

    # Safeguard against data loss
    if not output_path.endswith('.json'):
        output_path += '.json'

    # Write to file with error handling
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(animation_data, json_file, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to write animation data: {str(e)}") from e


def apply_savitzky_golay_smoothing(
        input_data: np.ndarray,
        window_length: int = 5,
        polyorder: int = 2,
        axis: int = 0,
        validate: bool = True
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Apply Savitzky-Golay filter smoothing along specified axis of input data.

    Args:
        input_data: 2D numpy array of shape (n_samples, n_features)
        window_length: Length of the filter window (must be odd and > polyorder)
        polyorder: Order of the polynomial fit
        axis: Axis along which to filter (0: column-wise, 1: row-wise)
        validate: Enable input validation checks when True

    Returns:
        tuple: (smoothed_data, processing_time)
               - smoothed_data: Smoothed output array
               - processing_time: Execution time in seconds (None in validation mode)

    Raises:
        ValueError: For invalid input dimensions or filter parameters
    """
    # Validation mode timing bypass
    processing_time = None

    if validate:
        # Input integrity checks
        if input_data.ndim != 2:
            raise ValueError(f"Expected 2D input, got {input_data.ndim}D array")

        if window_length % 2 == 0 or window_length < 3:
            raise ValueError("Window length must be odd integer ≥ 3")

        if polyorder >= window_length:
            raise ValueError("Polynomial order must be < window length")

    # Store original dtype and convert to float64 for numerical stability
    original_dtype = input_data.dtype
    working_data = input_data.astype(np.float64)

    # Start performance timer
    timer_start = time.perf_counter()

    try:
        # Vectorized Savitzky-Golay application
        smoothed_data = savgol_filter(working_data,
                                      window_length=window_length,
                                      polyorder=polyorder,
                                      axis=axis,
                                      mode='mirror')
    except Exception as e:
        raise RuntimeError(f"Filtering failed: {str(e)}") from e

    # Stop timer and calculate duration
    processing_time = time.perf_counter() - timer_start

    # Restore original data type with overflow protection
    return (
        np.clip(smoothed_data,
                0.0,
                1.0
                ).astype(original_dtype),
        processing_time
    )


def _blend_region_start(
    array: np.ndarray,
    region: np.ndarray,
    processed_boundary: int,
    blend_frames: int
) -> None:
    """Applies linear blend between last active frame and silent region start."""
    blend_length = min(blend_frames, region[0] - processed_boundary)
    if blend_length <= 0:
        return

    pre_frame = array[region[0] - 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[0] + i] = pre_frame * (1 - weight) + array[region[0] + i] * weight

def _blend_region_end(
    array: np.ndarray,
    region: np.ndarray,
    blend_frames: int
) -> None:
    """Applies linear blend between silent region end and next active frame."""
    blend_length = min(blend_frames, array.shape[0] - region[-1] - 1)
    if blend_length <= 0:
        return

    post_frame = array[region[-1] + 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[-1] - i] = post_frame * (1 - weight) + array[region[-1] - i] * weight

def find_low_value_regions(
        signal: np.ndarray,
        threshold: float,
        min_region_length: int = 5
) -> list:
    """Identifies contiguous regions in a signal where values fall below a threshold.

    Args:
        signal: Input 1D array of numerical values
        threshold: Value threshold for identifying low regions
        min_region_length: Minimum consecutive samples required to qualify as a region

    Returns:
        List of numpy arrays, each containing indices for a qualifying low-value region
    """
    low_value_indices = np.where(signal < threshold)[0]
    contiguous_regions = []
    current_region_length = 0
    region_start_idx = 0

    for i in range(1, len(low_value_indices)):
        # Check if current index continues a consecutive sequence
        if low_value_indices[i] != low_value_indices[i - 1] + 1:
            # Finalize previous region if it meets length requirement
            if current_region_length >= min_region_length:
                contiguous_regions.append(low_value_indices[region_start_idx:i])
            # Reset tracking for new potential region
            region_start_idx = i
            current_region_length = 0
        current_region_length += 1

    # Add the final region if it qualifies
    if current_region_length >= min_region_length:
        contiguous_regions.append(low_value_indices[region_start_idx:])

    return contiguous_regions


def smooth_mouth_movements(
        blend_shapes: np.ndarray,
        processed_frames: int,
        volume: np.ndarray = None,
        silence_threshold: float = 0.001,
        min_silence_duration: int = 7,
        blend_window: int = 3
) -> np.ndarray:
    """Reduces jaw movement artifacts during silent periods in audio-driven animation.

    Args:
        blend_shapes: Array of facial blend shape weights [num_frames, num_blendshapes]
        processed_frames: Number of already processed frames that shouldn't be modified
        volume: Audio volume array used to detect silent periods
        silence_threshold: Volume threshold for considering a frame silent
        min_silence_duration: Minimum consecutive silent frames to qualify for processing
        blend_window: Number of frames to smooth at region boundaries

    Returns:
        Modified blend shape array with reduced mouth movements during silence
    """
    if volume is None:
        return blend_shapes

    # Detect silence periods using volume data
    silent_regions = find_low_value_regions(
        volume,
        threshold=silence_threshold,
        min_region_length=min_silence_duration
    )

    for region_indices in silent_regions:
        # Reduce mouth blend shapes in silent region
        mouth_blend_indices = [ARKitBlendShape.index(name) for name in MOUTH_BLENDSHAPES]
        for region_indice in region_indices.tolist():
            blend_shapes[region_indice, mouth_blend_indices] *= 0.1

        try:
            # Smooth transition into silent region
            _blend_region_start(
                blend_shapes,
                region_indices,
                processed_frames,
                blend_window
            )

            # Smooth transition out of silent region
            _blend_region_end(
                blend_shapes,
                region_indices,
                blend_window
            )
        except IndexError as e:
            warnings.warn(f"Edge blending skipped at region {region_indices}: {str(e)}")

    return blend_shapes


def apply_frame_blending(
        blend_shapes: np.ndarray,
        processed_frames: int,
        initial_blend_window: int = 3,
        subsequent_blend_window: int = 5
) -> np.ndarray:
    """Smooths transitions between processed and unprocessed animation frames using linear blending.

    Args:
        blend_shapes: Array of facial blend shape weights [num_frames, num_blendshapes]
        processed_frames: Number of already processed frames (0 means no previous processing)
        initial_blend_window: Max frames to blend at sequence start
        subsequent_blend_window: Max frames to blend between processed and new frames

    Returns:
        Modified blend shape array with smoothed transitions
    """
    if processed_frames > 0:
        # Blend transition between existing and new animation
        _blend_animation_segment(
            blend_shapes,
            transition_start=processed_frames,
            blend_window=subsequent_blend_window,
            reference_frame=blend_shapes[processed_frames - 1]
        )
    else:
        # Smooth initial frames from neutral expression (zeros)
        _blend_animation_segment(
            blend_shapes,
            transition_start=0,
            blend_window=initial_blend_window,
            reference_frame=np.zeros_like(blend_shapes[0])
        )
    return blend_shapes


def _blend_animation_segment(
        array: np.ndarray,
        transition_start: int,
        blend_window: int,
        reference_frame: np.ndarray
) -> None:
    """Applies linear interpolation between reference frame and target frames.

    Args:
        array: Blend shape array to modify
        transition_start: Starting index for blending
        blend_window: Maximum number of frames to blend
        reference_frame: The reference frame to blend from
    """
    actual_blend_length = min(blend_window, array.shape[0] - transition_start)

    for frame_offset in range(actual_blend_length):
        current_idx = transition_start + frame_offset
        blend_weight = (frame_offset + 1) / (actual_blend_length + 1)

        # Linear interpolation: ref_frame * (1 - weight) + current_frame * weight
        array[current_idx] = (reference_frame * (1 - blend_weight)
                              + array[current_idx] * blend_weight)


BROW1 = np.array([[0.05597309, 0.05727929, 0.07995935, 0.        , 0.        ],
                   [0.00757574, 0.00936678, 0.12242376, 0.        , 0.        ],
                   [0.        , 0.        , 0.14943372, 0.04535687, 0.04264118],
                   [0.        , 0.        , 0.18015374, 0.09019445, 0.08736137],
                   [0.        , 0.        , 0.20549579, 0.12802747, 0.12450772],
                   [0.        , 0.        , 0.21098022, 0.1369939 , 0.13343132],
                   [0.        , 0.        , 0.20904602, 0.13903855, 0.13562402],
                   [0.        , 0.        , 0.20365039, 0.13977394, 0.13653506],
                   [0.        , 0.        , 0.19714841, 0.14096624, 0.13805152],
                   [0.        , 0.        , 0.20325482, 0.17303431, 0.17028868],
                   [0.        , 0.        , 0.21990852, 0.20164253, 0.19818163],
                   [0.        , 0.        , 0.23858181, 0.21908803, 0.21540019],
                   [0.        , 0.        , 0.2567876 , 0.23762083, 0.23396946],
                   [0.        , 0.        , 0.34093422, 0.27898848, 0.27651772],
                   [0.        , 0.        , 0.45288125, 0.35008961, 0.34887788],
                   [0.        , 0.        , 0.48076251, 0.36878952, 0.36778417],
                   [0.        , 0.        , 0.47798249, 0.36362219, 0.36145973],
                   [0.        , 0.        , 0.46186113, 0.33865979, 0.33597934],
                   [0.        , 0.        , 0.45264384, 0.33152157, 0.32891783],
                   [0.        , 0.        , 0.40986338, 0.29646468, 0.2945672 ],
                   [0.        , 0.        , 0.35628179, 0.23356403, 0.23155804],
                   [0.        , 0.        , 0.30870566, 0.1780673 , 0.17637439],
                   [0.        , 0.        , 0.25293985, 0.10710219, 0.10622486],
                   [0.        , 0.        , 0.18743332, 0.03252602, 0.03244236],
                   [0.02340254, 0.02364671, 0.15736724, 0.        , 0.        ]])

BROW2 = np.array([
                [0.        , 0.        , 0.09799323, 0.05944436, 0.05002545],
                [0.        , 0.        , 0.09780276, 0.07674237, 0.01636653],
                [0.        , 0.        , 0.11136199, 0.1027964 , 0.04249811],
                [0.        , 0.        , 0.26883412, 0.15861984, 0.15832305],
                [0.        , 0.        , 0.42191629, 0.27038204, 0.27007768],
                [0.        , 0.        , 0.3404977 , 0.21633868, 0.21597538],
                [0.        , 0.        , 0.27301185, 0.17176409, 0.17134669],
                [0.        , 0.        , 0.25960442, 0.15670464, 0.15622253],
                [0.        , 0.        , 0.22877269, 0.11805892, 0.11754539],
                [0.        , 0.        , 0.1451605 , 0.06389034, 0.0636282 ]])

BROW3 = np.array([
                [0.    , 0.    , 0.124 , 0.0295, 0.0295],
                [0.    , 0.    , 0.267 , 0.184 , 0.184 ],
                [0.    , 0.    , 0.359 , 0.2765, 0.2765],
                [0.    , 0.    , 0.3945, 0.3125, 0.3125],
                [0.    , 0.    , 0.4125, 0.331 , 0.331 ],
                [0.    , 0.    , 0.4235, 0.3445, 0.3445],
                [0.    , 0.    , 0.4085, 0.3305, 0.3305],
                [0.    , 0.    , 0.3695, 0.294 , 0.294 ],
                [0.    , 0.    , 0.2835, 0.213 , 0.213 ],
                [0.    , 0.    , 0.1795, 0.1005, 0.1005],
                [0.    , 0.    , 0.108 , 0.014 , 0.014 ]])


import numpy as np
from scipy.ndimage import label


def apply_random_brow_movement(input_exp, volume):
    FRAME_SEGMENT = 150
    HOLD_THRESHOLD = 10
    VOLUME_THRESHOLD = 0.08
    MIN_REGION_LENGTH = 6
    STRENGTH_RANGE = (0.7, 1.3)

    BROW_PEAKS = {
        0: np.argmax(BROW1[:, 2]),
        1: np.argmax(BROW2[:, 2])
    }

    for seg_start in range(0, len(volume), FRAME_SEGMENT):
        seg_end = min(seg_start + FRAME_SEGMENT, len(volume))
        seg_volume = volume[seg_start:seg_end]

        candidate_regions = []

        high_vol_mask = seg_volume > VOLUME_THRESHOLD
        labeled_array, num_features = label(high_vol_mask)

        for i in range(1, num_features + 1):
            region = (labeled_array == i)
            region_indices = np.where(region)[0]
            if len(region_indices) >= MIN_REGION_LENGTH:
                candidate_regions.append(region_indices)

        if candidate_regions:
            selected_region = candidate_regions[np.random.choice(len(candidate_regions))]
            region_start = selected_region[0]
            region_end = selected_region[-1]
            region_length = region_end - region_start + 1

            brow_idx = np.random.randint(0, 2)
            base_brow = BROW1 if brow_idx == 0 else BROW2
            peak_idx = BROW_PEAKS[brow_idx]

            if region_length > HOLD_THRESHOLD:
                local_max_pos = seg_volume[selected_region].argmax()
                global_peak_frame = seg_start + selected_region[local_max_pos]

                rise_anim = base_brow[:peak_idx + 1]
                hold_frame = base_brow[peak_idx:peak_idx + 1]

                insert_start = max(global_peak_frame - peak_idx, seg_start)
                insert_end = min(global_peak_frame + (region_length - local_max_pos), seg_end)

                strength = np.random.uniform(*STRENGTH_RANGE)

                if insert_start + len(rise_anim) <= seg_end:
                    input_exp[insert_start:insert_start + len(rise_anim), :5] += rise_anim * strength
                    hold_duration = insert_end - (insert_start + len(rise_anim))
                    if hold_duration > 0:
                        input_exp[insert_start + len(rise_anim):insert_end, :5] += np.tile(hold_frame * strength,
                                                                                           (hold_duration, 1))
            else:
                anim_length = base_brow.shape[0]
                insert_pos = seg_start + region_start + (region_length - anim_length) // 2
                insert_pos = max(seg_start, min(insert_pos, seg_end - anim_length))

                if insert_pos + anim_length <= seg_end:
                    strength = np.random.uniform(*STRENGTH_RANGE)
                    input_exp[insert_pos:insert_pos + anim_length, :5] += base_brow * strength

    return np.clip(input_exp, 0, 1)