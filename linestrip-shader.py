#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
import sys
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / (right - left)
    M[3, 0] = -(right + left) / float(right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[3, 1] = -(top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M

vertex_code = """
#version 120
uniform mat4 projection;
uniform float linelength;
uniform float antialias;
uniform float linewidth;

attribute vec3 prev;
attribute vec3 curr;
attribute vec3 next;

varying vec2 v_texcoord;
void main()
{
    float z[3] = float[](1,2,3);

    float w = linewidth/2.0 + 1.5*antialias;
    float dy = w;
    vec2 miter;

    float sign = +1.0;
    if(curr.z < 0.0)
    {
        sign = -1.0;
    }

    // Start of line
    if( curr.xy == prev.xy )
    {
        vec2 v = normalize(next.xy - curr.xy);
        miter = vec2(-v.y, v.x);
    }
    // End of line
    else if ( curr.xy == next.xy )
    {
        vec2 v = normalize(curr.xy - prev.xy);
        miter = vec2(-v.y, v.x);
    }
    // Regular segment
    else
    {
        vec2 v0 = normalize(curr.xy - prev.xy);
        vec2 v1 = normalize(next.xy - curr.xy);
        vec2 n0 = vec2(-v0.y, v0.x);
        vec2 n1 = vec2(-v1.y, v1.x);
        miter = normalize(n0 + n1);
        dy = w / dot(miter, n1);
    }
    vec2 p = curr.xy + dy * sign * miter;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_texcoord = vec2((abs(curr.z)-1)/linelength, sign*w);
} """

fragment_code = """
#version 120
uniform float antialias;
uniform float linewidth;
varying vec2 v_texcoord;

void main()
{
    float distance = v_texcoord.y;
    float d = abs(distance) - linewidth/2.0 + antialias;
    float alpha = 1.0;
    if( d > 0.0 )
    {
        alpha = d/(antialias);
        alpha = exp(-alpha*alpha);
    }
    gl_FragColor = vec4(0, 0, 0, alpha); // * v_texcoord.x);
} """


# GLUT callbacks
# --------------------------------------
def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 2, len(data)-4)
    glut.glutSwapBuffers()

def reshape(width,height):
    gl.glViewport(0, 0, width, height)
    projection = ortho(0, width, 0, height, -100, 100)
    loc = gl.glGetUniformLocation(program, "projection")
    gl.glUniformMatrix4fv(loc, 1, False, projection)

def keyboard( key, x, y ):
    if key == '\033':
        sys.exit( )

# GLUT init
# --------------------------------------
glut.glutInit(sys.argv)
# HiDPI support for retina display
# This requires glut from http://iihm.imag.fr/blanch/software/glut-macosx/
if sys.platform == 'darwin':
    import ctypes
    from OpenGL import platform
    try:
        glutInitDisplayString = platform.createBaseFunction(
            'glutInitDisplayString', dll=platform.GLUT, resultType=None,
            argTypes=[ctypes.c_char_p],
            doc='glutInitDisplayString(  ) -> None',
        argNames=() )
        # text = ctypes.c_char_p("rgba stencil double samples=8 hidpi")
        text = ctypes.c_char_p("rgba double hidpi")
        glutInitDisplayString(text)
    except:
        pass
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutCreateWindow('Antialiased lines using fragment shader')
glut.glutReshapeWindow(800,800)
glut.glutReshapeFunc(reshape)
glut.glutDisplayFunc(display)
glut.glutKeyboardFunc(keyboard)

def star(inner=0.5, outer=1.0, n=5):
    R = np.array( [inner,outer]*n)
    T = np.linspace(0,2*np.pi,2*n,endpoint=False)
    P = np.zeros((2*n,2))
    P[:,0]= R*np.cos(T)
    P[:,1]= R*np.sin(T)
    return P

# Data & parameters
# --------------------------------------
def bake(data):
    """
    (n,2) vertices -> (4,3) + 2*(n,3) vertices
    2*n floats     -> 12 + 6*n floats
    """
    # Prepare space
    V = np.zeros( (2+2+2*len(data),3), dtype=np.float32)

    # Double each vertex (required for thick lines)
    V[2:-2:2, 0:2] = V[3:-2:2, 0:2] = data

    # Put extra vertex at start and end
    V[ 0,:2] = V[ 1,:2] = data[0]
    V[-2,:2] = V[-1,:2] = data[-1]

    # Compute distance (optional)
    P = V[2:-2:2,:2]
    D = ((P[:-1]-P[1:])**2).sum(axis=1)
    # We store 1 + actual length such that we encode sign
    # within the distance. In fragment shader, curr.z-1 must be used
    V[4:-2:2,2] = V[5:-2:2,2] = 1+np.sqrt(D).cumsum()
    V[-2:,2] = V[-3,2]
    length = V[-3,2]

    # Disambiguate vertices
    V[1::2,2] *= -1

    return V, length

antialias = 1.0

# linewidth = 35.0
# line = [(100,400), (400,600), (700,400)]

# Nice spiral
n = 1024
linewidth = 1.0
T = np.linspace(0, 12*2*np.pi, n)
R = np.linspace(10, 400, n)
line = np.zeros((n,2), dtype=np.float32)
line[:,0] = 400 + np.cos(T)*R
line[:,1] = 400 + np.sin(T)*R

# Star
# n = 5
# linewidth = 20.0
# line = (star(n=n)*400 + (400,400))

# Baking data
data,length = bake(line)


# Build & activate program
# --------------------------------------
program  = gl.glCreateProgram()

vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
gl.glShaderSource(vertex, vertex_code)
gl.glCompileShader(vertex)
gl.glAttachShader(program, vertex)

fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
gl.glShaderSource(fragment, fragment_code)
gl.glCompileShader(fragment)
gl.glAttachShader(program, fragment)

gl.glLinkProgram(program)
gl.glDetachShader(program, vertex)
gl.glDetachShader(program, fragment)
gl.glUseProgram(program)


# Build buffer & bind attributes
# --------------------------------------
vbuffer = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbuffer)
gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
stride = data.strides[0]
offset = ctypes.c_void_p(0)

loc = gl.glGetAttribLocation(program, "curr")
gl.glEnableVertexAttribArray(loc)
gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

loc = gl.glGetAttribLocation(program, "prev")
gl.glEnableVertexAttribArray(loc)
offset = ctypes.c_void_p(-2*stride)
gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

loc = gl.glGetAttribLocation(program, "next")
gl.glEnableVertexAttribArray(loc)
offset = ctypes.c_void_p(+2*stride)
gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

loc = gl.glGetUniformLocation(program, "linewidth")
gl.glUniform1f(loc, linewidth)
loc = gl.glGetUniformLocation(program, "antialias")
gl.glUniform1f(loc, antialias)
loc = gl.glGetUniformLocation(program, "linelength")
gl.glUniform1f(loc, length)

# OpenGL initalization
# --------------------------------------
gl.glClearColor(1,1,1,1)
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

# Enter mainloop
# --------------------------------------
glut.glutMainLoop()
