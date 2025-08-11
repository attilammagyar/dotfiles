Dotfiles
========

My dotfiles and helper scripts for Vim, Bash, Screen, etc.

Files and directories
---------------------

 * `dotfiles`: my configuration files for CLI tools
    * `bashrc`
    * `screenrc`
    * `vimrc`
    * `gitconfig`
    * `gitignore`
 * `scripts`: helper scripts for `dotfiles` and automations of repetitive tasks
    * `ai.py`: standalone CLI tool for chatting with various LLMs (work-in-progress)
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
