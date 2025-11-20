This README file will feature a simple workflow for you to get a better understanding of the project. And then, all functionalities, parameters will explained throughly.

1-Create a virtual environment. "python -m venv .venv"
2-Activate the virtual environment. ".venv/Scripts/Activate"
3-Check src/utils. First function is there to split the training data (rests in "data/dataset")
into training, validation and testing.
4-Check src/dataloader. This creates 3 dataset objects to load .wav files under the given folder. Train has shuffle attribute set to true while val and test are set to false. The reason for this is to make each epoch see the samples in a random order while keeping val and test consistent.
5-Check src/model. This is used to create the model itself. Many intricacies. Try to grasp this part with the help of AI. A mel spectogram snippet is provided below in this file.

    A mel spectrogram is like an image:
    Height = frequencies (mel bins)
    Width = time
    Pixel color = energy of that frequency at that time
    Examples of patterns in spectrograms:
    A steady tone → a horizontal line
    A drum hit → a vertical line
    A melody → a curved pattern
    Chords/harmony → stacked lines
    CNNs are really good at detecting these patterns.

