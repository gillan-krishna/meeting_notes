# Meeting notes

# Before you use
First install dependencies using requirements.txt file \
Create an OpenAI account (to use the summarization) https://beta.openai.com/account/api-keys \
Copy and keep the API key \
\
Store your audio file in audio files folder \
Change path of audio file in transcription.py\
Enter number of speakers, language ('English' or 'any'),  in setEnv.py\
Run transcription.py, and wait for a while for transcription\
\
After transcription is done, press any key to identify speakers\
After audio is played, Enter name of identified speaker. If audio is not clear, enter 0 to play another clip\
After identifying all speakers, you transcription file is ready as transript.txt

# Summarization
To include a short summary, first enter OpenAI account key in main function summarization.py
Run summarization.py to get transcript with a summary on top as summary.txt
