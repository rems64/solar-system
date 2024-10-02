main: glad
	g++ src/main.cpp lib/glad.o -lglfw -lGL -g -I include/ -o bin/main

glad:
	g++ src/glad.c -I include/ -c -o lib/glad.o

# sphere_shader:
# 	cp shaders/sphere.vert bin/shaders/sphere.vert
# 	cp shaders/sphere.frag bin/shaders/sphere.frag

run: bin/main
	./bin/main
