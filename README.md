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
 * `scripts`: helper scripts for `dotfiles` and automations of repetitive tasks
    * `backup.sh`: create an encrypted and gzipped tarball from a directory
    * `restore.sh`: restore a directory from a backup created by `backup.sh`
    * `ctags.sh`: `ctags` helper script for my `.vimrc`
    * `lookup_docs.sh`: API documentation finder for my `.vimrc`
    * `vid2android.sh`: convert and resize videos for my android phone (2.2)
    * `bake`: make-shift wrapper around make to do compiling and test
              installations out-of-tree
    * `md2bloghtml`: convert markdown files to HTML to be used in my company
                     blog
    * `mntprvt`: mount the encrypted part of my home directory
    * `wifi.sh`: connect to wi-fi from console, without `network-manager`
