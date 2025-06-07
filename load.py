import glob
import os
import pickle


def load_pickle_file(filename=None):
    if filename is None:
        files = glob.glob("*gameplay*.pkl")
        if not files:
            print("No gameplay files found!")
            return None

        # Get the newest file
        filename = max(files, key=os.path.getctime)
        print(f"Loading newest file: {filename}")

    print(f"File size: {os.path.getsize(filename):,} bytes")

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded successfully!")
        print(f"Data type: {type(data)}")

        if isinstance(data, list):
            print(f"Number of frames: {len(data)}")
            if len(data) > 0:
                print(f"First frame type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First frame keys: {list(data[0].keys())}")

        return data

    except Exception as e:
        print(f"Failed to load: {e}")
        return None


def validate_training_data(data):
    if not data:
        print("No data to validate")
        return False

    if not isinstance(data, list):
        print("Data is not a list")
        return False

    valid_frames = 0
    for frame in data:
        if (isinstance(frame, dict) and
                'state' in frame and
                frame['state'] is not None and
                'action' in frame):
            valid_frames += 1

    print(f"Valid training frames: {valid_frames}/{len(data)}")

    if valid_frames > 50:
        print("Data looks good for training!")
        return True
    else:
        print("Not enough valid frames for training")
        return False


def main():
    print("Gameplay Data Loader")
    print("=" * 30)

    data = load_pickle_file()

    if data:
        if validate_training_data(data):
            print("\nData ready for training!")
            print("Run: python train.py")
        else:
            print("\nData has issues but was loaded successfully")
    else:
        print("\nCould not load any files")


if __name__ == "__main__":
    main()
