#!/bin/sh
set -e
cd "$(dirname "$0")"

git submodule update --init

mkdir -p textures
u=https://www.solarsystemscope.com/textures/download

echo "Downloading any missing texture..."
for f in \
  2k_sun.jpg \
  2k_earth_daymap.jpg \
  2k_earth_nightmap.jpg \
  2k_mercury.jpg \
  2k_venus_surface.jpg \
  2k_mars.jpg \
  2k_moon.jpg \
  2k_stars.jpg
do
  if [ ! -f "textures/$f" ]; then
    curl -fL "$u/$f" -o "textures/$f"
  fi
done

for f in 2k_earth_normal_map 2k_earth_specular_map
do
  if [ ! -f "textures/$f.png" ]; then
    if [ ! -f "textures/$f.tif" ]; then
      curl -fL "$u/$f.tif" -o "textures/$f.tif"
    fi
    ffmpeg -y -loglevel error -i "textures/$f.tif" "textures/$f.png"
  fi
done

echo "Setup is done!"
