import subprocess
import os


def install_requirements():
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')

def create_config_env():
    api_key = input("Please enter your LLM_API_KEY: ")
    api_url = input("Please enter your LLM_BASE_URL: ")
    config_content = f"LLM_API_KEY={api_key}\nLLM_BASE_URL={api_url}\n"
    with open("config.env", "w") as f:
        f.write(config_content)
    print("config.env has been created.")

if __name__ == "__main__":
    print("Setup started.")
    install_requirements()
    create_config_env()
    os.makedirs("data/raw", exist_ok=True)
    print("Setup complete. You can proceed now.")