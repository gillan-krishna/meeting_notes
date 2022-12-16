import openai

def Summarize(path):
    text = open(path, 'r').read()
    text = 'Summarize the following conversation: ' + text
    response = openai.Completion.create(engine='text-davinci-003', prompt=text, temperature=0.7)
    return response['choices'][0]['text']

    # print(response)

if __name__ == '__main__':
    openai.api_key = 'Enter your API key here'
    path = 'transcript.txt'
    summary = Summarize(path)

    text_to_prepend = summary

    with open(path, "r") as file:
        # Read the existing text from the file
        existing_text = file.read()

    with open("summary.txt", "w") as file:
        # Write the new text followed by the existing text
        file.write('Summary:')
        file.write(text_to_prepend + '\n\n')
        file.write('Details:')
        file.write(existing_text)
