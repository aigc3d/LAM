from pathlib import Path
from typing import Optional
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from vhap.util.log import get_logger


logger = get_logger(__name__)


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        image_folder: Path,
        background_folder: Optional[Path]=None,
        background_fname2camId=lambda x: x,
        image_fname2camId=lambda x: x,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <image_folder>/
                |---xx.jpg
                |---...
        """
        super().__init__()
        self.image_fname2camId = image_fname2camId
        self.background_foler = background_folder

        logger.info(f"Initializing dataset from folder {image_folder}")

        self.image_paths = sorted(list(image_folder.glob('*.jpg')))

        if background_folder is not None:
            self.backgrounds = {}
            background_paths = sorted(list((image_folder / background_folder).glob('*.jpg')))

            for background_path in background_paths:
                bg = np.array(Image.open(background_path))
                cam_id = background_fname2camId(background_path.name)
                self.backgrounds[cam_id] = bg
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        cam_id = self.image_fname2camId(image_path.name)
        rgb = np.array(Image.open(image_path))
        item = {
            "rgb": rgb,
            'image_path': str(image_path),
        }

        if self.background_foler is not None:
            item['background'] = self.backgrounds[cam_id]

        return item


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = ImageFolderDataset(
        image_folder='./xx',
        img_to_tensor=True,
    )

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())
    print(sample["rgb"].shape)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    for item in tqdm(dataloader):
        pass
