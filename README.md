Dotfiles
========

My dotfiles and helper scripts for [Vim](https://www.vim.org/),
[Bash](https://www.gnu.org/software/bash/),
[Screen](https://www.gnu.org/software/screen/), etc.

Files and directories
---------------------

 * `dotfiles`: my configuration files for CLI tools
    * `bashrc`
    * `screenrc`
    * `vimrc`
    * `gitconfig`
    * `gitignore`
 * `scripts`: helper scripts for `dotfiles` and automations of repetitive tasks
    * `ai.py`: standalone CLI tool for chatting with various LLMs (beware the
      sass though)
    * `backup.sh`: create an encrypted and gzipped tarball from a directory
    * `restore.sh`: restore a directory from a backup created by `backup.sh`
    * `bake`: make-shift wrapper around make to do compiling and test
              installations out-of-tree
    * `loudness.sh`: print loudness statistics for audio files using ffmpeg
    * `mntprvt`: mount the encrypted part of my home directory
    * `vim_ctags.sh`: `ctags` helper script for my `.vimrc`
    * `vim_git.sh`: `git` helper script for my `.vimrc`
    * `vim_lookup_docs.sh`: API documentation finder for my `.vimrc`
    * `vim_run_tests.sh`: test runner script for my `.vimrc`
    * `wifi.sh`: connect to wi-fi from console, without `network-manager`
    * `gitgrepfight.sh`: compare the number of matches of regular expressions
                         inside a `git` repository
 * `etc`: global configuration files
   * `hosts`: slightly modified version of the hosts file available at
     http://someonewhocares.org/hosts/
   * `iptables-rules`: firewall rules
   * `iptables-rules-vps`: firewall rules on my VPS with NAT for VPN

ai.py
-----

`ai.py` is a command-line API client application to access a few popular
[LLM](https://en.wikipedia.org/wiki/Large_language_model) providers.

Its features include:

 * A system prompt which sets up a sassy, wisecracking AI assistant that has a
   knack for programming and [STEM](https://en.wikipedia.org/wiki/Science,_technology,_engineering,_and_mathematics)
   problem solving.

 * Precise control over model selection,
   [sampling temperature](https://en.wikipedia.org/wiki/Softmax_function),
   reasoning, etc.

 * Allows editing the entire conversation in Markdown, including the AI's
   responses. (Useful for nudging the autoregressive text generation process.)

 * Works as a standalone interactive CLI app or as a standard Unix filter that
   can be integrated with editors like Vim.

 * Can connect to the API of:

    * [Anthropic](https://www.anthropic.com/),
    * [DeepSeek](https://www.deepseek.com/en),
    * [Google](https://gemini.google.com/),
    * [OpenAI](https://openai.com/),
    * and [Perplexity](https://www.perplexity.ai/).

### ai.py as an interactive CLI app

<img src="https://raw.githubusercontent.com/attilammagyar/dotfiles/main/images/ai-py-interactive.gif" alt="ai.py running as an interactive CLI app" />

### ai.py as a standard Unix filter integrated into Vim

<img src="https://raw.githubusercontent.com/attilammagyar/dotfiles/main/images/ai-py-vim.gif" alt="ai.py integrated into Vim" />

### Setting up ai.py

To use `ai.py`, you need to generate an API key for at least one of the
supported AI providers, and save it in `~/.ai-py` in the following format
(delete the ones that you don't want to use):

    {
      "api_keys": {
        "anthropic": "Anthropic Claude API key here (https://console.anthropic.com/settings/keys)",
        "deepseek": "DeepSeek R1 API key here (https://platform.deepseek.com/api_keys)",
        "google": "Google Gemini API key here (https://aistudio.google.com/apikey)",
        "openai": "OpenAI ChatGPT API key here (https://platform.openai.com/settings/organization/api-keys)",
        "perplexity": "Perplexity API key here (https://www.perplexity.ai/account/api/keys)"
      }
    }

For the Vim integration, see the `AIFilter` function in my
[.vimrc](https://github.com/attilammagyar/dotfiles/blob/main/dotfiles/vimrc)
file.
