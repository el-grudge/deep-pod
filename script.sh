#!/bin/bash

# Download SQLite 3.45.3 or the desired version
wget https://www.sqlite.org/2023/sqlite-autoconf-3450300.tar.gz
tar -xvzf sqlite-autoconf-3450300.tar.gz

# Build and install the downloaded SQLite version
cd sqlite-autoconf-3450300
./configure --prefix=$HOME/.local
make
make install

# Add SQLite to the PATH so it's available for your app
export PATH="$HOME/.local/bin:$PATH"
