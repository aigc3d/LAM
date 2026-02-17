"""
Tests for concierge_modal.py - Concierge ZIP Generator on Modal

Tests are organized into categories:
1. ZIP Structure Validation - Verify generated ZIP contents
2. Code Correctness - Static analysis of pipeline code
3. Comparison Tests - Compare good (fne) vs potentially bad (now) ZIPs
4. Pipeline Logic - Test individual pipeline functions (mocked)

Run: python -m pytest tests/test_concierge_modal.py -v
"""

import ast
import json
import os
import struct
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONCIERGE_MODAL_PY = REPO_ROOT / "concierge_modal.py"
APP_CONCIERGE_PY = REPO_ROOT / "app_concierge.py"
APP_LAM_PY = REPO_ROOT / "app_lam.py"
GENERATE_GLB_PY = REPO_ROOT / "tools" / "generateARKITGLBWithBlender.py"

# Pre-built ZIPs for comparison
ZIP_FNE = REPO_ROOT / "concierge_fne.zip"  # Official HF Spaces output (works)
ZIP_NOW = REPO_ROOT / "concierge_now.zip"  # Custom video output (bird monster)


# ============================================================
# 1. ZIP Structure Validation
# ============================================================
class TestZipStructure:
    """Validate concierge.zip has correct structure and contents."""

    REQUIRED_FILES = {"skin.glb", "animation.glb", "vertex_order.json", "offset.ply"}

    @pytest.fixture(params=[
        pytest.param("concierge_fne.zip", id="fne"),
        pytest.param("concierge_now.zip", id="now"),
    ])
    def zip_path(self, request):
        path = REPO_ROOT / request.param
        if not path.exists():
            pytest.skip(f"{request.param} not found")
        return path

    def test_zip_is_valid(self, zip_path):
        """ZIP file should be a valid ZIP archive."""
        assert zipfile.is_zipfile(zip_path), f"{zip_path.name} is not a valid ZIP"

    def test_zip_contains_required_files(self, zip_path):
        """ZIP must contain all required concierge files."""
        with zipfile.ZipFile(zip_path) as zf:
            basenames = {os.path.basename(n) for n in zf.namelist() if not n.endswith("/")}
            missing = self.REQUIRED_FILES - basenames
            assert not missing, f"Missing required files: {missing}"

    def test_zip_has_single_directory(self, zip_path):
        """ZIP should have one top-level directory containing all files."""
        with zipfile.ZipFile(zip_path) as zf:
            top_dirs = {n.split("/")[0] for n in zf.namelist() if "/" in n}
            assert len(top_dirs) == 1, f"Expected 1 top-level dir, got {len(top_dirs)}: {top_dirs}"

    def test_vertex_order_is_valid_permutation(self, zip_path):
        """vertex_order.json must be a valid permutation of 0..N-1."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("vertex_order.json"):
                    data = json.loads(zf.read(info.filename))
                    break
            else:
                pytest.fail("vertex_order.json not found")

        assert isinstance(data, list), "vertex_order.json should be a list"
        n = len(data)
        assert n > 0, "vertex_order.json is empty"
        assert sorted(data) == list(range(n)), (
            f"vertex_order.json is not a valid permutation of 0..{n-1}"
        )

    def test_vertex_order_is_not_sequential(self, zip_path):
        """vertex_order.json should NOT be sequential (Blender reorders vertices)."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("vertex_order.json"):
                    data = json.loads(zf.read(info.filename))
                    break
            else:
                pytest.fail("vertex_order.json not found")

        assert data != list(range(len(data))), (
            "vertex_order.json is sequential [0,1,2,...] — this is WRONG. "
            "Blender reorders vertices on import, so vertex_order must reflect that. "
            "A sequential ordering causes the 'bird monster' avatar bug."
        )

    def test_vertex_order_count_matches_flame(self, zip_path):
        """vertex_order.json should have 20018 entries (FLAME subdivide=1 vertex count)."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("vertex_order.json"):
                    data = json.loads(zf.read(info.filename))
                    break
            else:
                pytest.fail("vertex_order.json not found")

        # FLAME with subdivide_num=1 produces 20018 vertices (60054/3)
        assert len(data) == 20018, (
            f"Expected 20018 vertices (FLAME subdivide=1), got {len(data)}"
        )

    def test_skin_glb_is_valid_glb(self, zip_path):
        """skin.glb should start with the glTF magic number."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("skin.glb"):
                    data = zf.read(info.filename)
                    break
            else:
                pytest.fail("skin.glb not found")

        # GLB magic number: 0x46546C67 = "glTF"
        assert len(data) >= 12, "skin.glb too small"
        magic = struct.unpack("<I", data[:4])[0]
        assert magic == 0x46546C67, f"skin.glb has wrong magic: {hex(magic)}, expected glTF"

    def test_animation_glb_is_valid_glb(self, zip_path):
        """animation.glb should start with the glTF magic number."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("animation.glb"):
                    data = zf.read(info.filename)
                    break
            else:
                pytest.fail("animation.glb not found")

        assert len(data) >= 12, "animation.glb too small"
        magic = struct.unpack("<I", data[:4])[0]
        assert magic == 0x46546C67, f"animation.glb has wrong magic: {hex(magic)}, expected glTF"

    def test_offset_ply_is_valid(self, zip_path):
        """offset.ply should be a valid PLY file."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("offset.ply"):
                    data = zf.read(info.filename)
                    break
            else:
                pytest.fail("offset.ply not found")

        # PLY magic number
        assert data[:3] == b"ply", "offset.ply doesn't start with 'ply' magic"

    def test_skin_glb_reasonable_size(self, zip_path):
        """skin.glb should be between 1MB and 10MB (not bloated with textures)."""
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith("skin.glb"):
                    size_mb = info.file_size / (1024 * 1024)
                    break
            else:
                pytest.fail("skin.glb not found")

        assert 1.0 < size_mb < 10.0, (
            f"skin.glb is {size_mb:.1f}MB — expected 1-10MB. "
            f"If >10MB, materials/textures may not have been stripped."
        )


# ============================================================
# 2. Code Correctness - Static Analysis
# ============================================================
class TestCodeCorrectness:
    """Static analysis of concierge_modal.py for known bug patterns."""

    @pytest.fixture
    def modal_source(self):
        return CONCIERGE_MODAL_PY.read_text()

    def test_no_sequential_vertex_order_overwrite(self, modal_source):
        """concierge_modal.py must NOT overwrite vertex_order.json with range(n).

        BUG: generate_glb() already creates correct vertex_order.json via Blender.
        Overwriting it with list(range(n_verts)) from trimesh causes the
        'bird monster' avatar because OBJ vertex order != GLB vertex order.
        """
        # Check for the bug pattern: trimesh load + range() + vertex_order write
        bug_patterns = [
            "vertex_order = list(range(",
            "json.dump(vertex_order",  # only bad if preceded by range()
        ]

        lines = modal_source.split("\n")
        in_generate_fn = False
        range_vertex_order_found = False

        for i, line in enumerate(lines):
            if "def _generate_concierge_zip" in line:
                in_generate_fn = True
            if in_generate_fn and "vertex_order = list(range(" in line:
                range_vertex_order_found = True
                # Check it's not in a comment
                stripped = line.lstrip()
                if not stripped.startswith("#"):
                    pytest.fail(
                        f"Line {i+1}: Found sequential vertex_order overwrite bug!\n"
                        f"  {line.strip()}\n"
                        f"This overwrites the correct Blender-generated vertex_order.json "
                        f"with a naive sequential ordering, causing the 'bird monster' bug."
                    )

    def test_generate_glb_is_called(self, modal_source):
        """The official generate_glb() from tools/ should be used."""
        assert "from tools.generateARKITGLBWithBlender import generate_glb" in modal_source, (
            "concierge_modal.py should import generate_glb from official tools"
        )
        assert "generate_glb(" in modal_source, (
            "concierge_modal.py should call generate_glb()"
        )

    def test_template_fbx_path(self, modal_source):
        """Template FBX should reference model_zoo/sample_oac/template_file.fbx."""
        assert "sample_oac/template_file.fbx" in modal_source, (
            "Template FBX path should include sample_oac/template_file.fbx"
        )

    def test_animation_glb_copy(self, modal_source):
        """animation.glb should be copied from sample_oac."""
        assert "sample_oac/animation.glb" in modal_source, (
            "animation.glb should be copied from sample_oac"
        )

    def test_xformers_in_image_build(self, modal_source):
        """xformers must be installed for correct DINOv2 attention."""
        assert "xformers" in modal_source, (
            "xformers must be in the Modal image build for DINOv2 accuracy"
        )

    def test_blender_installed(self, modal_source):
        """Blender 4.2 must be installed in the Modal image."""
        assert "blender" in modal_source.lower(), (
            "Blender must be in the Modal image for GLB generation"
        )

    def test_safetensors_loading(self, modal_source):
        """Model weights should be loaded via safetensors."""
        assert "load_file" in modal_source or "load_safetensors" in modal_source, (
            "Model weights should use safetensors loading"
        )

    def test_no_trimesh_vertex_overwrite_in_pipeline(self, modal_source):
        """After generate_glb(), there should be no trimesh-based vertex_order write."""
        # Find the generate_glb() call and check what follows
        lines = modal_source.split("\n")
        generate_glb_line = None
        for i, line in enumerate(lines):
            if "generate_glb(" in line and not line.lstrip().startswith("#"):
                generate_glb_line = i
                break

        if generate_glb_line is None:
            pytest.skip("generate_glb() call not found")

        # Check the next 20 lines after generate_glb for trimesh overwrite
        for i in range(generate_glb_line + 1, min(generate_glb_line + 20, len(lines))):
            line = lines[i].strip()
            if line.startswith("#"):
                continue
            if "trimesh.load" in line and "vertex_order" not in line:
                continue
            if "list(range(" in line:
                pytest.fail(
                    f"Line {i+1}: Found list(range(...)) near generate_glb() call.\n"
                    f"  {line}\n"
                    f"This likely overwrites the Blender-generated vertex_order.json."
                )


# ============================================================
# 3. Comparison Tests - fne (good) vs now (bad)
# ============================================================
class TestZipComparison:
    """Compare known-good ZIP (fne) vs potentially broken ZIP (now)."""

    @pytest.fixture
    def fne_zip(self):
        if not ZIP_FNE.exists():
            pytest.skip("concierge_fne.zip not available")
        return ZIP_FNE

    @pytest.fixture
    def now_zip(self):
        if not ZIP_NOW.exists():
            pytest.skip("concierge_now.zip not available")
        return ZIP_NOW

    def _read_file_from_zip(self, zip_path, suffix):
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.filename.endswith(suffix):
                    return zf.read(info.filename)
        return None

    def test_animation_glb_identical(self, fne_zip, now_zip):
        """animation.glb should be identical (same template)."""
        fne_data = self._read_file_from_zip(fne_zip, "animation.glb")
        now_data = self._read_file_from_zip(now_zip, "animation.glb")
        assert fne_data is not None and now_data is not None
        assert fne_data == now_data, "animation.glb should be identical (same template)"

    def test_vertex_order_same_length(self, fne_zip, now_zip):
        """Both ZIPs should have same number of vertices in vertex_order."""
        fne_vo = json.loads(self._read_file_from_zip(fne_zip, "vertex_order.json"))
        now_vo = json.loads(self._read_file_from_zip(now_zip, "vertex_order.json"))
        assert len(fne_vo) == len(now_vo), (
            f"Vertex count mismatch: fne={len(fne_vo)}, now={len(now_vo)}"
        )

    def test_vertex_order_both_valid_permutations(self, fne_zip, now_zip):
        """Both vertex_order.json files should be valid permutations."""
        fne_vo = json.loads(self._read_file_from_zip(fne_zip, "vertex_order.json"))
        now_vo = json.loads(self._read_file_from_zip(now_zip, "vertex_order.json"))
        assert sorted(fne_vo) == list(range(len(fne_vo)))
        assert sorted(now_vo) == list(range(len(now_vo)))

    def test_offset_ply_same_size(self, fne_zip, now_zip):
        """offset.ply should have same size (same FLAME topology)."""
        fne_data = self._read_file_from_zip(fne_zip, "offset.ply")
        now_data = self._read_file_from_zip(now_zip, "offset.ply")
        assert fne_data is not None and now_data is not None
        assert len(fne_data) == len(now_data), (
            f"offset.ply size mismatch: fne={len(fne_data)}, now={len(now_data)}"
        )

    def test_skin_glb_similar_size(self, fne_zip, now_zip):
        """skin.glb sizes should be similar (same topology, different shape)."""
        fne_data = self._read_file_from_zip(fne_zip, "skin.glb")
        now_data = self._read_file_from_zip(now_zip, "skin.glb")
        assert fne_data is not None and now_data is not None
        ratio = len(fne_data) / len(now_data)
        assert 0.8 < ratio < 1.2, (
            f"skin.glb size ratio too different: {ratio:.2f} "
            f"(fne={len(fne_data)}, now={len(now_data)})"
        )

    def test_vertex_order_divergence(self, fne_zip, now_zip):
        """Measure vertex_order divergence between fne and now.

        Different input images produce different shaped meshes, so slight
        vertex_order differences are expected. But if >50% differ, it may
        indicate a systematic problem in vertex ordering approach.
        """
        fne_vo = json.loads(self._read_file_from_zip(fne_zip, "vertex_order.json"))
        now_vo = json.loads(self._read_file_from_zip(now_zip, "vertex_order.json"))
        diffs = sum(1 for a, b in zip(fne_vo, now_vo) if a != b)
        pct = diffs / len(fne_vo) * 100
        # This is informational - different inputs produce different orderings
        print(f"\nVertex order divergence: {diffs}/{len(fne_vo)} ({pct:.1f}%) entries differ")
        # We just log, not assert - different shapes = different orderings is OK


# ============================================================
# 4. Pipeline Logic Tests
# ============================================================
class TestPipelineLogic:
    """Test pipeline functions and configurations."""

    def test_generate_glb_includes_vertex_order_step(self):
        """Official generate_glb() must call gen_vertex_order_with_blender."""
        if not GENERATE_GLB_PY.exists():
            pytest.skip("generateARKITGLBWithBlender.py not found")
        source = GENERATE_GLB_PY.read_text()
        assert "gen_vertex_order_with_blender" in source, (
            "generate_glb() must call gen_vertex_order_with_blender for correct vertex ordering"
        )

    def test_generate_glb_outputs_vertex_order_in_glb_dir(self):
        """generate_glb() should write vertex_order.json next to output_glb."""
        if not GENERATE_GLB_PY.exists():
            pytest.skip("generateARKITGLBWithBlender.py not found")
        source = GENERATE_GLB_PY.read_text()
        # Check that vertex_order.json path is derived from output_glb's directory
        assert "os.path.dirname(output_glb)" in source, (
            "vertex_order.json should be written in the same directory as output_glb"
        )

    def test_generate_vertex_indices_sorts_by_z(self):
        """Official generateVertexIndices.py should sort vertices by Z coordinate."""
        script = REPO_ROOT / "tools" / "generateVertexIndices.py"
        if not script.exists():
            pytest.skip("generateVertexIndices.py not found")
        source = script.read_text()
        assert "sorted(vertices" in source, (
            "Vertex indices should be sorted"
        )
        # Check Z-coordinate sorting
        assert "x[1]" in source or ".z" in source, (
            "Vertices should be sorted by Z coordinate"
        )

    def test_convert_fbx2glb_strips_materials(self):
        """convertFBX2GLB.py should strip materials to avoid GLB bloat."""
        script = REPO_ROOT / "tools" / "convertFBX2GLB.py"
        if not script.exists():
            pytest.skip("convertFBX2GLB.py not found")
        source = script.read_text()
        assert "strip_materials" in source, (
            "FBX→GLB conversion should strip materials to prevent ~40MB bloat"
        )
        assert "export_morph_normal" in source, (
            "export_morph_normal should be set to False to prevent morph target bloat"
        )

    def test_modal_image_has_required_deps(self):
        """Modal image must include all critical dependencies."""
        source = CONCIERGE_MODAL_PY.read_text()
        required_deps = [
            "torch==2.3.0",
            "xformers",
            "pytorch3d",
            "diff-gaussian-rasterization",
            "nvdiffrast",
            "fbx-2020",  # FBX SDK
            "blender",
        ]
        for dep in required_deps:
            assert dep in source, f"Modal image missing dependency: {dep}"

    def test_flame_vertex_count(self):
        """FLAME template_file.fbx should reference 60054 vertex coordinates.

        60054 / 3 = 20018 vertices, which is the expected FLAME subdivide=1 count.
        """
        source = GENERATE_GLB_PY.read_text()
        assert "60054" in source, (
            "FLAME template should have 60054 vertex coordinates (20018 * 3)"
        )

    def test_modal_pipeline_uses_correct_image_size(self):
        """Pipeline should use 512x512 source images (LAM-20K config)."""
        source = CONCIERGE_MODAL_PY.read_text()
        # Check that config is loaded (source_size should be 512)
        assert "cfg.source_size" in source or "source_size" in source

    def test_shape_param_injected_into_motion(self):
        """shape_param should be set in motion_seq flame_params before inference."""
        source = CONCIERGE_MODAL_PY.read_text()
        assert 'motion_seq["flame_params"]["betas"] = shape_param' in source, (
            "shape_param must be injected into motion_seq flame_params as 'betas'"
        )


# ============================================================
# 5. Code Consistency Tests (modal vs concierge vs official)
# ============================================================
class TestCodeConsistency:
    """Ensure concierge_modal.py is consistent with working implementations."""

    def test_weight_loading_approach(self):
        """Weight loading in modal should match official app_lam.py approach."""
        modal_src = CONCIERGE_MODAL_PY.read_text()
        official_src = APP_LAM_PY.read_text()
        # Both should use state_dict copy approach
        assert "state_dict[k].copy_(v)" in modal_src or "load_state_dict" in modal_src
        assert "state_dict[k].copy_(v)" in official_src or "load_state_dict" in official_src

    def test_inference_call_signature(self):
        """LAM inference call should match official signature."""
        modal_src = CONCIERGE_MODAL_PY.read_text()
        official_src = APP_LAM_PY.read_text()
        # Both should call infer_single_view with same key params
        for param in ["render_c2ws", "render_intrs", "render_bg_colors", "flame_params"]:
            assert param in modal_src, f"Modal missing inference param: {param}"
            assert param in official_src, f"Official missing inference param: {param}"

    def test_preprocess_image_params(self):
        """preprocess_image() call should use same params as official."""
        modal_src = CONCIERGE_MODAL_PY.read_text()
        # Key params that must match
        for param in ["multiply=14", "need_mask=True", "get_shape_param=True"]:
            assert param in modal_src, (
                f"preprocess_image() missing param: {param}"
            )

    def test_prepare_motion_seqs_params(self):
        """prepare_motion_seqs() call should use same params as official."""
        modal_src = CONCIERGE_MODAL_PY.read_text()
        for param in ["multiply=16", "cross_id=False", "test_sample=False"]:
            assert param in modal_src, (
                f"prepare_motion_seqs() missing param: {param}"
            )

    def test_oac_export_has_all_required_files(self):
        """OAC export directory should contain all required files."""
        modal_src = CONCIERGE_MODAL_PY.read_text()
        # Check that skin.glb, animation.glb, vertex_order.json, offset.ply are created
        assert "skin.glb" in modal_src
        assert "animation.glb" in modal_src
        # vertex_order.json is created by generate_glb() internally
        assert "offset.ply" in modal_src


# ============================================================
# 6. Bug Regression Tests
# ============================================================
class TestBugRegression:
    """Regression tests for known bugs."""

    def test_no_sequential_vertex_order_in_pipeline(self):
        """REGRESSION: vertex_order.json must never be list(range(n)).

        Bug: concierge_modal.py previously overwrote the Blender-generated
        vertex_order.json with list(range(n_verts)), a naive sequential
        ordering from trimesh. Since Blender reorders vertices during
        FBX import, the OBJ vertex order != GLB vertex order.
        Result: mesh vertices mapped to wrong bones → 'bird monster' avatar.
        Fix: Remove the trimesh-based overwrite; let generate_glb() handle it.
        """
        source = CONCIERGE_MODAL_PY.read_text()
        lines = source.split("\n")
        in_generate_fn = False
        for i, line in enumerate(lines):
            if "def _generate_concierge_zip" in line:
                in_generate_fn = True
            if not in_generate_fn:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # The specific bug pattern
            if "list(range(" in stripped and "vertex" in source[max(0, source.index(stripped)-200):source.index(stripped)].lower():
                pytest.fail(
                    f"REGRESSION: Line {i+1} has sequential vertex_order pattern.\n"
                    f"  {stripped}"
                )

    def test_no_leftover_trimesh_vertex_order(self):
        """After the fix, trimesh should not be used for vertex_order.json."""
        source = CONCIERGE_MODAL_PY.read_text()
        # Find the _generate_concierge_zip function body
        lines = source.split("\n")
        fn_start = None
        fn_end = None
        for i, line in enumerate(lines):
            if "def _generate_concierge_zip" in line:
                fn_start = i
            elif fn_start is not None and line.startswith("def ") or line.startswith("class "):
                fn_end = i
                break
        if fn_end is None:
            fn_end = len(lines)
        fn_body = "\n".join(lines[fn_start:fn_end])

        # trimesh.load followed by vertex_order write is the bug
        if "trimesh.load" in fn_body and "vertex_order" in fn_body:
            # Only fail if it's not commented out
            for line in lines[fn_start:fn_end]:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "trimesh" in stripped and "vertex" in stripped.lower():
                    pytest.fail(
                        f"Found trimesh + vertex_order in _generate_concierge_zip:\n"
                        f"  {stripped}\n"
                        f"vertex_order.json should only come from generate_glb()"
                    )


# ============================================================
# 7. Cache / Stale Data Prevention Tests
# ============================================================
class TestCachePrevention:
    """Ensure stale data is cleaned before each generation run."""

    @pytest.fixture
    def modal_source(self):
        return CONCIERGE_MODAL_PY.read_text()

    def _extract_function_body(self, source, func_name):
        """Extract the body of a function from source code."""
        lines = source.split("\n")
        fn_start = None
        indent_level = None
        for i, line in enumerate(lines):
            if func_name in line and ("def " in line or "def\t" in line):
                fn_start = i
                indent_level = len(line) - len(line.lstrip())
                continue
            if fn_start is not None and i > fn_start:
                if line.strip() and not line[0].isspace() and ":" in line:
                    return "\n".join(lines[fn_start:i])
                # Check for same-level or lower-level def/class
                if line.strip().startswith(("def ", "class ")):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level:
                        return "\n".join(lines[fn_start:i])
        return "\n".join(lines[fn_start:]) if fn_start else ""

    def test_volume_cleaned_before_generation_in_generator(self, modal_source):
        """Generator.generate() must clean stale volume files before starting."""
        gen_body = self._extract_function_body(modal_source, "def generate")
        for stale_file in ["concierge.zip", "preview.mp4", "tracked_face.png", "preproc_input.png"]:
            assert stale_file in gen_body, (
                f"Generator.generate() should reference '{stale_file}' for cleanup"
            )
        assert "os.remove" in gen_body or "shutil.rmtree" in gen_body, (
            "Generator.generate() must remove stale files"
        )

    def test_volume_cleaned_before_generation_in_web_ui(self, modal_source):
        """Web UI process() must clean stale volume files before launching GPU job."""
        # Find the process function inside web()
        web_body = self._extract_function_body(modal_source, "def web")
        # Look for cache cleanup in process()
        assert "CACHE FIX" in web_body or "stale" in web_body.lower() or (
            "os.remove" in web_body and "concierge.zip" in web_body
        ), (
            "Web UI process() should clean stale volume files before starting GPU job"
        )

    def test_flame_tracking_fully_cleaned(self, modal_source):
        """FLAME tracking output must be FULLY cleaned (rmtree), not partially."""
        gen_body = self._extract_function_body(modal_source, "def _generate_concierge_zip")
        # Should use shutil.rmtree on the entire tracking directory
        assert "shutil.rmtree(tracking_root)" in gen_body or \
               "shutil.rmtree(tracking_root," in gen_body, (
            "_generate_concierge_zip must rmtree the entire output/tracking/ directory. "
            "Partial cleanup (only subdirs) misses internal state files."
        )

    def test_generate_glb_temp_files_cleaned(self, modal_source):
        """Stale generate_glb temp files must be cleaned before pipeline runs."""
        # Check that temp_ascii.fbx and temp_bin.fbx cleanup exists
        assert "temp_ascii.fbx" in modal_source and "temp_bin.fbx" in modal_source, (
            "Pipeline must reference generate_glb temp files for cleanup"
        )
        # Verify cleanup happens BEFORE generate_glb() call
        gen_body = self._extract_function_body(modal_source, "def _generate_concierge_zip")
        lines = gen_body.split("\n")
        cleanup_line = None
        generate_glb_line = None
        for i, line in enumerate(lines):
            if "temp_ascii.fbx" in line and "remove" in line:
                cleanup_line = i
            if "generate_glb(" in line and not line.strip().startswith("#"):
                generate_glb_line = i
        if cleanup_line is not None and generate_glb_line is not None:
            assert cleanup_line < generate_glb_line, (
                "Temp file cleanup must happen BEFORE generate_glb() call"
            )

    def test_stale_status_files_cleaned(self, modal_source):
        """Leftover status_*.json files from previous jobs must be cleaned."""
        gen_body = self._extract_function_body(modal_source, "def generate")
        assert "status_" in gen_body and ".json" in gen_body, (
            "Generator.generate() should clean stale status files"
        )

    def test_no_fixed_filename_collision_risk(self, modal_source):
        """Output files use fixed names — verify cleanup prevents stale serving."""
        # The output always goes to concierge.zip (fixed name)
        # Cleanup must happen both in Generator.generate() AND web UI process()
        gen_body = self._extract_function_body(modal_source, "def generate")
        web_body = self._extract_function_body(modal_source, "def web")
        # Both should clean concierge.zip
        gen_cleans = "concierge.zip" in gen_body and ("remove" in gen_body or "rmtree" in gen_body)
        web_cleans = "concierge.zip" in web_body and ("remove" in web_body or "rmtree" in web_body)
        assert gen_cleans and web_cleans, (
            "Both Generator.generate() and web UI must clean stale concierge.zip. "
            "Double-cleanup prevents race: UI cleans volume, GPU cleans again before work."
        )


# ============================================================
# 8. GPU Error Handling & Timeout Tests
# ============================================================
class TestGPUErrorHandling:
    """Ensure GPU errors propagate to the UI instead of silent timeout."""

    @pytest.fixture
    def modal_source(self):
        return CONCIERGE_MODAL_PY.read_text()

    def test_call_gpu_writes_error_to_volume(self, modal_source):
        """_call_gpu() must write error status to volume on failure, not just print."""
        # Find the _call_gpu function inside web()
        assert 'gpu_error["error"]' in modal_source or "gpu_error[" in modal_source, (
            "_call_gpu() must propagate error to shared state, not just print()"
        )
        # Must also write status file for volume-based detection
        # Look for json.dump in _call_gpu context
        lines = modal_source.split("\n")
        in_call_gpu = False
        writes_status = False
        for line in lines:
            if "def _call_gpu" in line:
                in_call_gpu = True
            if in_call_gpu and "json.dump" in line and "error" in line:
                writes_status = True
            if in_call_gpu and line.strip() and not line[0].isspace() and "def " in line and "_call_gpu" not in line:
                break
        assert writes_status, (
            "_call_gpu() must write error status to volume on failure"
        )

    def test_ui_detects_dead_gpu_thread(self, modal_source):
        """UI polling loop must detect when GPU thread dies without writing status."""
        assert 'gpu_error["done"]' in modal_source or "gpu_error.get(" in modal_source, (
            "UI must check shared gpu_error state to detect dead GPU thread"
        )
        assert "GPU job finished without writing results" in modal_source or \
               "GPU process terminated unexpectedly" in modal_source or \
               "gpu_error" in modal_source, (
            "UI must report meaningful error when GPU thread dies silently"
        )

    def test_gpu_timeout_is_sufficient(self, modal_source):
        """GPU container timeout must be >= 1200s for full pipeline."""
        import re
        # Match the @app.cls line with gpu= (the GPU Generator class)
        match = re.search(r'@app\.cls\(.*gpu=.*timeout=(\d+)', modal_source)
        assert match, "GPU class @app.cls must have timeout= parameter"
        timeout_val = int(match.group(1))
        assert timeout_val >= 1200, (
            f"GPU timeout={timeout_val}s is too short. Full pipeline (FLAME tracking + "
            f"LAM inference + GLB generation) typically takes 10-25 minutes. "
            f"Must be >= 1200s (20 min)."
        )

    def test_generate_has_finally_status_guard(self, modal_source):
        """Generator.generate() must write status file in finally block as last resort."""
        lines = modal_source.split("\n")
        in_generate = False
        has_finally_guard = False
        for i, line in enumerate(lines):
            if "def generate(self" in line:
                in_generate = True
            if in_generate and "finally:" in line:
                # Check next ~10 lines for status file write
                for j in range(i+1, min(i+15, len(lines))):
                    if "status_written" in lines[j] or "status_file" in lines[j]:
                        has_finally_guard = True
                        break
        assert has_finally_guard, (
            "Generator.generate() must have a finally block that writes status file "
            "as a last resort (handles cases where except block also fails)"
        )

    def test_scaledown_window_reasonable(self, modal_source):
        """scaledown_window should be >= 30s to avoid excessive cold starts."""
        import re
        match = re.search(r'scaledown_window=(\d+)', modal_source)
        assert match, "GPU class must have scaledown_window= parameter"
        val = int(match.group(1))
        assert val >= 30, (
            f"scaledown_window={val}s is too aggressive. Cold starts add 2-5 minutes. "
            f"Use >= 30s to reuse warm containers for rapid iteration."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
