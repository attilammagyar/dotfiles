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
    * `backup.sh`: create an encrypted and gzipped tarball from a directory
    * `restore.sh`: restore a directory from a backup created by `backup.sh`
    * `vid2android.sh`: convert and resize videos for my android phone (2.2)
    * `bake`: make-shift wrapper around make to do compiling and test
              installations out-of-tree
    * `md2bloghtml`: convert markdown files to HTML to be used in my company
                     blog
    * `mntprvt`: mount the encrypted part of my home directory
    * `vim_ctags.sh`: `ctags` helper script for my `.vimrc`
    * `vim_git.sh`: `git` helper script for my `.vimrc`
    * `vim_lookup_docs.sh`: API documentation finder for my `.vimrc`
    * `vim_run_tests.sh`: test runner script for my `.vimrc`
    * `update_ci_screenshots.sh`: update Jenkins or Travis CI screenshots for
                                  screensaver
    * `webshot.py`: CLI script to take a screenshot from a webpage URL
    * `wifi.sh`: connect to wi-fi from console, without `network-manager`
    * `gitgrepfight.sh`: compare the number of matches of regular expressions
                         inside a `git` repository
 * `etc`: global configuration files
   * `iptables-rules`: firewall rules
   * `iptables-rules-vps`: firewall rules on my VPS with NAT for VPN
