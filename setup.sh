#!/bin/bash

DOTFILES_REPO=$(dirname "$0")

mkdir -vp ~/bin
cp -vr $DOTFILES_REPO/scripts/* ~/bin/
chmod +x ~/bin/*

cp -v $DOTFILES_REPO/dotfiles/bashrc ~/.my_bashrc
echo "source ~/.my_bashrc" >> ~/.bashrc

cp -v $DOTFILES_REPO/dotfiles/vimrc ~/.vimrc

cp -v $DOTFILES_REPO/dotfiles/screenrc ~/.screenrc

mkdir -vp ~/.ssh
cp -v $DOTFILES_REPO/dotfiles/ssh/config ~/.ssh/config

cp -v $DOTFILES_REPO/dotfiles/xinitrc ~/.xinitrc

mkdir -vp ~/.fluxbox
cp -vr $DOTFILES_REPO/dotfiles/fluxbox/* ~/.fluxbox/

cp -v dotfiles/gitignore ~/.gitignore
cp -v dotfiles/gitconfig ~/.gitconfig

sudo cp -v $DOTFILES_REPO/etc/iptables-rules /etc/
sudo bash -c "cat '$DOTFILES_REPO/etc/hosts' >> /etc/hosts"
sudo mkdir -vp /etc/network/if-pre-up.d
sudo bash -c \
    'echo -e "#!/bin/bash\\n\\n/sbin/iptables-restore < /etc/iptables-rules" \
    > /etc/network/if-pre-up.d/iptables'
sudo chmod +x /etc/network/if-pre-up.d/iptables

cd /tmp/
wget -O Monaco_Linux.ttf 'http://www.gringod.com/wp-upload/software/Fonts/Monaco_Linux.ttf'
sudo mkdir -p /usr/share/fonts/truetype/custom
sudo mv Monaco_Linux.ttf /usr/share/fonts/truetype/custom/
sudo fc-cache -f -v
