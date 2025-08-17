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
[LLM](https://en.wikipedia.org/wiki/Large_language_model) providers solving
most of my complaints against existing tools (see below).

Its features include:

 * A system prompt which sets up a sassy, wisecracking AI assistant that has a
   knack for programming and [STEM](https://en.wikipedia.org/wiki/Science,_technology,_engineering,_and_mathematics)
   problem solving.

 * Precise control over model selection,
   [sampling temperature](https://en.wikipedia.org/wiki/Softmax_function),
   reasoning, etc. (Can switch models even in the middle of a conversation.)

 * Allows editing the entire conversation, including the AI's responses.
   (Useful for steering and nudging the autoregressive text generation
   process.)

 * Simple Markdown-based syntax.

 * Works as a standalone interactive CLI app or as a standard Unix filter
   that can be integrated with editors like Vim.

 * Can connect to the API of:

    * [Anthropic](https://www.anthropic.com/),
    * [DeepSeek](https://www.deepseek.com/en),
    * [Google](https://gemini.google.com/),
    * [OpenAI](https://openai.com/),
    * [Perplexity](https://www.perplexity.ai/),
    * and [xAI](https://x.ai/).

### Demo

`ai.py` running as an **interactive CLI** app (`ai.py interactive` - the
default when the standard input is a TTY):

<img src="https://raw.githubusercontent.com/attilammagyar/dotfiles/main/images/ai-py-interactive.gif" alt="ai.py running as an interactive CLI app" />

`ai.py` running as a **standard Unix filter** (`ai.py stdio` - the default when
the standard input is not a TTY), integrated into Vim:

<img src="https://raw.githubusercontent.com/attilammagyar/dotfiles/main/images/ai-py-vim.gif" alt="ai.py integrated into Vim" />

### Dependencies

None. Only built-in Python modules.

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
        "perplexity": "Perplexity API key here (https://www.perplexity.ai/account/api/keys)",
        "xai": "xAI API key here (https://console.x.ai/team/default/api-keys)"
      }
    }

For the Vim integration, see the `AIFilter` function in my
[.vimrc](https://github.com/attilammagyar/dotfiles/blob/main/dotfiles/vimrc)
file.

Alternatively, you can leave the `api_keys` field empty in `~/.ai-py`, and
provide the API keys via the following environment variables as well:

 * `ANTHROPIC_API_KEY`,
 * `DEEPSEEK_API_KEY`,
 * `GEMINI_API_KEY`,
 * `OPENAI_API_KEY`,
 * `PERPLEXITY_API_KEY`,
 * `XAI_API_KEY`.

### Syntax

A basic conversation after a few turns may look like this:

    # === System ===

    Please act as a helpful AI assistant.

    # === Settings ===

    Model: openai/gpt-4.1
    Reasoning: default
    Streaming: on
    Temperature: 1.0

    # === User ===

    Please explain why the sky is blue.

    # === AI ===

    Because the air scatters the blue light more than the other colors.

    # === User ===

    Say that again but talk like a pirate.

    # === AI ===

    Yarr, 'tis 'cause th' air be scatterin' th' blue light more'n all th' other
    colors, savy?

The `System` block contains general behavioural instructions for the model,
a.k.a. the system prompt. It is always automatically moved to the beginning of
the conversation, and when multiple `System` blocks are specified, only the
last one is kept. When it is omitted, then the built-in system prompt from
`ai.py` is used.

When the last block in a conversation is a `User` block, then `ai.py` will send
the conversation to the selected LLM and generate a response for it.

If the optional `Settings` block or any of the settings in it are omitted, then
the values from the last `ai.py` interaction are used. Subsequent `Setting`
blocks and settings overwrite each other.

`ai.py` also adds additional information blocks to the conversation:

 * `Notes`: a few tips for using `ai.py`, and a complete list of the available
   models from the configured providers.

 * `AI Reasoning`: the chain-of-thought tokens or summaries generated by large
    reasoning models.

 * `AI Status`: token usage and other API-specific status info.

These are included only for convenience and as diagnostic information, but are
never sent back to the AI.

If `ai.py` is used as a Unix filter and its standard input is empty, then it
will generate an empty conversation template, including its default system
prompt.

### Why?

It's intended to solve my complaints against existing solutions:

1. I'm not a fan of digital feudalism and the subscription based economy, and I
   avoid it like the plague. Prepaid API credits are somewhat closer to my
   heart.

2. GUIs and agents: limited to no control over model selection, system prompt,
   etc.

3. Agents: if I have to keep an eye on what they are doing, then they don't
   seem to be that much of a speed up. But I'm also not comfortable with
   leaving them to work alone, changing files and running arbitrary commands
   based on a stochastic process, regardless of how sophisticated that may be.

4. You get good at what you practice, and you lose skills that you don't.
   Therefore I try to limit my AI usage to only the boring parts of programming
   (e.g. cookie-cutter UI stuff, boilerplate, etc.), and to get my work
   reviewed.

5. There is undeniably a hype in the industry right now (as of August, 2025).
   When the music stops and the VC money dries up, several products will go
   down, others will get heavily "enshittified". Developing a reliance on any
   particular service risks future pain.

6. AFAIK the legal status of AI generated code is still unclear (as of August,
   2025): can you put large chunks of it in open source (e.g. GPL) or
   proprietary projects?  How much do you have to modify it before you can?
   Where's the threshold of originality?  Do you have to include the prompts as
   well?

7. The default AI persona drives me nuts with its excessive sugarcoating and
   politeness.

8. I've switched to Vim around 2011 and never looked back. IDEs came and went
   over the years (Eclipse, NetBeans, IntelliJ, Sublime, VS Code, etc.), but so
   far nothing has beaten Vim in terms of value vs. cost, especially with its
   stellar scriptability and macros. I don't see a reason for this to change
   any time soon.
