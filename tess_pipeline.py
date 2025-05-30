import os
import shutil
import random

from config import TRAINING_FILES_PATH
from config import TESS_ORIGINAL_FOLDER_PATH


class TESSPipeline:

    @staticmethod
    def create_tess_folders(path):
        """
        We are filling folders Actor_25 if YAF and Actor_26 if OAF.
        Example filename: 03-01-07-02-02-01-01.wav
        """
        counter = 0

        label_conversion = {'01': 'neutral',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fear',
                            '07': 'disgust',
                            '08': 'ps'}

        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if filename.startswith('OAF'):
                    destination_path = TRAINING_FILES_PATH + 'Actor_26\\'
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate base from extension
                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if base.endswith(value):
                            random_list = random.sample(range(10, 99), 7)
                            file_name = '-'.join([str(i) for i in random_list])
                            file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)

                else:
                    destination_path = TRAINING_FILES_PATH + 'Actor_25\\'
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate base from extension
                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if base.endswith(value):
                            random_list = random.sample(range(10, 99), 7)
                            file_name = '-'.join([str(i) for i in random_list])
                            file_name_with_correct_emotion = (file_name[:6] + key + file_name[8:] + extension).strip()
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)


if __name__ == '__main__':
    TESSPipeline.create_tess_folders(TESS_ORIGINAL_FOLDER_PATH)
