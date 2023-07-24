from utils import create_input_files

if __name__ == '__main__':

    create_input_files(dataset='flicker8k',
                           karpathy_json_path='D:\pyproject\pose2.5k\pose2.5k.json',
                           image_folder='D:\pyproject\pose2.5k\pose_image',
                           captions_per_image=1,
                           min_word_freq=0,
                           output_folder='D:\pyproject\pose2.5k\output',
                           max_len=50)
