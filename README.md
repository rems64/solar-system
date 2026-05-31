# Solar system

This project was partially developed in the context of an IGR course in Telecom Paris

This is a simple solar system featuring :
- Bloom
- Earth atmosphere shader
- Dead ImGUI for controls
- Smooth camera motions

## Setup

```bash
./setup.sh
make
make run
```

Textures are downloaded from [Solar System Scope](https://www.solarsystemscope.com/textures/) (CC BY 4.0). Earth normal/specular maps are converted from TIFF to PNG with ffmpeg.

## LSP support

To generate `compile_commands.json`, run :

```bash
make lsp
```

## Screenshots

![Earth's view](/imgs/earth_view.png)
![Solar system](/imgs/solar_system.png)
![Earth's atmosphere](/imgs/atmosphere.png)
