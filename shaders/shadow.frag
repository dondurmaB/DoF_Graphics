#version 330 core

// No color output: the shadow FBO has only a depth attachment (see
// resizeShadowFramebuffer in main.cpp). gl_FragDepth is written implicitly
// from gl_Position.z, so this stage needs no body.
void main()
{
}
