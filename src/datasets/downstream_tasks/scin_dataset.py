import os
import pandas as pd
from PIL import Image
from ....src.datasets.base_dataset import BaseDataset

class SCINDataset(BaseDataset):
    """
    PyTorch Dataset for the SCIN (Skin Condition Image Network) dataset.
    Each sample returns:
      - 'case_id': string
      - 'images': list of PIL Images
      - 'shot_types': list of shot-type strings
      - 'metadata': dict of all available fields
      - 'labels': the weighted differential label dict
      - 'description': continuous-text description of each image
    """

    IMG_COL = "path"
    LBL_COL ='weighted_skin_condition_label'

    def __init__(
        self,
        root: str,
        cases_file: str = "scin_cases.csv",
        labels_file: str = "scin_labels.csv",
        transform=None,
        val_transform=None,
        **kwargs,
    ):
        super().__init__(transform=transform, val_transform=val_transform, **kwargs)
        self.root = root

        # Load metadata CSVs
        cases_df = pd.read_csv(os.path.join(root, cases_file))
        labels_df = pd.read_csv(os.path.join(root, labels_file))
        self.meta_data = pd.merge(cases_df, labels_df, on="case_id", how="left")
        self.meta_data['path'] = self.meta_data['image_1_path'].apply(lambda x: str(self.root.parent / x))

        # Generate continuous-text descriptions for each row
        def _join_flags(row, prefix):
            items = [col.replace(prefix, "").lower().replace("_", " ")
                     for col, val in row.items() if col.startswith(prefix) and val]
            return ", ".join(items) if items else "none"

        def _safe_str(val):
            return str(val).lower() if isinstance(val, str) else "unknown"

        def _generate_description(row):
            shot = _safe_str(row.get(f"image_1_shot_type", "unknown"))
            sex = _safe_str(row.get('sex_at_birth'))
            age = str(row.get('age_group')) if pd.notna(row.get('age_group')) else "unknown"
            fitz = str(row.get('fitzpatrick_skin_type')) if pd.notna(row.get('fitzpatrick_skin_type')) else "unknown"

            race = _join_flags(row, "race_ethnicity_")
            location = _join_flags(row, "body_parts_")
            texture = _join_flags(row, "textures_")
            symptoms = _join_flags(row, "condition_symptoms_")

            others = [col.replace("other_symptoms_", "").lower().replace("_", " ")
                      for col, val in row.items() if col.startswith("other_symptoms_") and val]
            other_txt = f"; also {', '.join(others)}" if others else ""

            fst_labels = [row[f"dermatologist_fitzpatrick_skin_type_label_{j}"]
                          for j in (1, 2, 3)
                          if pd.notna(row.get(f"dermatologist_fitzpatrick_skin_type_label_{j}"))]
            fst_text = ", ".join(map(str, fst_labels)) if fst_labels else "n/a"

            grad_key = f"dermatologist_gradable_for_skin_condition_1"
            gradable = str(row.get(grad_key, False)).lower()

            label_name = str(row.get('dermatologist_skin_condition_label_name'))
            confidence = str(row.get('dermatologist_skin_condition_confidence'))
            weighted = row.get('weighted_skin_condition_label')
            duration = str(row.get('condition_duration')) if pd.notna(row.get('condition_duration')) else "unknown"
            monk_india = str(row.get('monk_skin_tone_label_india')) if pd.notna(row.get('monk_skin_tone_label_india')) else "n/a"
            monk_us = str(row.get('monk_skin_tone_label_us')) if pd.notna(row.get('monk_skin_tone_label_us')) else "n/a"

            para = (
                f"This clinical image is a {shot} view. "
                f"The patient is {sex} in the {age} age group with Fitzpatrick type {fitz}. "
                f"They reported race/ethnicity as {race}. "
                f"The lesion appears on {location} and is described as {texture}. "
                f"They experienced {symptoms}{other_txt} over {duration}. "
                f"A dermatologist marked this image gradable: {gradable}. "
                f"They labelled the condition {label_name} (confidence {confidence}); "
                f"the weighted differential is {weighted}. "
                f"Retrospective Fitzpatrick estimates by the dermatologist are {fst_text}, "
                f"and lay graders assigned monk skin tone {monk_india} (India) and {monk_us} (US)."
            )
            para = para.replace('_', ' ')
            return para

        self.meta_data['description'] = self.meta_data.apply(lambda row: _generate_description(row), axis=1)
        self.case_ids = self.meta_data['case_id'].tolist()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]

        # Load images
        images, shot_types = [], []
        for i in range(1, 4):
            path_key = f"image_{i}_path"
            img_path = row.get(path_key)
            if isinstance(img_path, str) and os.path.isfile(os.path.join(self.root, img_path)):
                img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                shot_types.append(row.get(f"image_{i}_shot_type"))

        sample = {
            'case_id': row['case_id'],
            'images': images,
            'shot_types': shot_types,
            'metadata': {col: row[col] for col in self.meta_data.columns if col not in ['case_id']},
            'labels': row.get('weighted_skin_condition_label'),
            'description': row['description']
        }
        return sample
