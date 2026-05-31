CXXFLAGS = -g
GLFLAGS = -lglfw -lGL

CP = cp
RM = rm
MKDIR = mkdir -p

LIBAVCODECS = -lavcodec -lavformat -lavutil -lswscale

BUILDDIR = build
SRCDIR = src
LIBDIR = lib
IMGUI_DIR = $(LIBDIR)/imgui
SHADERS_PREFIX = $(BUILDDIR)/shaders/

SHADERS_NAMES = atmosphere.frag base.vert fullscreen.vert compositing.frag shading.frag sun.frag textured.frag bloom_downsample.frag bloom_upsample.frag presentation.frag
SHADERS = $(addprefix $(SHADERS_PREFIX),$(SHADERS_NAMES))

IMGUI_SRC = imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp backends/imgui_impl_opengl3.cpp backends/imgui_impl_glfw.cpp
IMGUI_OBJ = $(addprefix $(BUILDDIR)/,$(patsubst %.cpp,%.o,$(subst backends/,,$(IMGUI_SRC))))

INCLUDES = -Iinclude -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
LSP_CXXFLAGS = -std=c++17 -DGLFW_INCLUDE_NONE $(INCLUDES)
LSP_CFLAGS = -Iinclude

all: $(BUILDDIR)/main

$(BUILDDIR)/main: $(SRCDIR)/main.cpp $(BUILDDIR)/glad.o $(IMGUI_OBJ) | $(SHADERS) $(BUILDDIR)
	$(CXX) $(SRCDIR)/main.cpp $(BUILDDIR)/glad.o $(IMGUI_OBJ) $(GLFLAGS) $(LIBAVCODECS) $(CXXFLAGS) $(INCLUDES) -o $@

$(BUILDDIR)/glad.o: $(SRCDIR)/glad.c | $(BUILDDIR)
	$(CXX) $(SRCDIR)/glad.c $(LSP_CFLAGS) -c -o $@

$(BUILDDIR)/%.o: $(IMGUI_DIR)/%.cpp | $(BUILDDIR)
	$(CXX) $^ -c -I $(IMGUI_DIR)/ -o $@

$(BUILDDIR)/%.o: $(IMGUI_DIR)/backends/%.cpp | $(BUILDDIR)
	$(CXX) $^ -c -I $(IMGUI_DIR)/ -I $(IMGUI_DIR)/backends/ -o $@

$(BUILDDIR)/shaders/%.vert: shaders/%.vert | $(BUILDDIR)/shaders
	$(CP) shaders/$*.vert $@

$(BUILDDIR)/shaders/%.frag: shaders/%.frag | $(BUILDDIR)/shaders
	$(CP) shaders/$*.frag $@

$(BUILDDIR)/shaders: | $(BUILDDIR)
	$(MKDIR) $@

$(BUILDDIR):
	$(MKDIR) $@

run: $(BUILDDIR)/main
	./$(BUILDDIR)/main

clean::
	@$(RM) -rf $(BUILDDIR)

.PHONY: lsp
lsp: compile_commands.json

compile_commands.json: Makefile
	@printf '[\n' > $@
	@printf '  {"directory":"%s","file":"src/main.cpp","command":"clang++ %s -c src/main.cpp"},\n' '$(CURDIR)' '$(LSP_CXXFLAGS)' >> $@
	@printf '  {"directory":"%s","file":"src/glad.c","command":"clang %s -c src/glad.c"}\n' '$(CURDIR)' '$(LSP_CFLAGS)' >> $@
	@printf ']\n' >> $@
