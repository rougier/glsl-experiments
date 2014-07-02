#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
# This implements antialiased lines using a geometry shader with correct joins
# and caps.
#
#
#
#
# -----------------------------------------------------------------------------
import sys
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from OpenGL.GL.EXT.geometry_shader4 import *

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
attribute vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
} """

fragment_code = """
#version 120
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;

varying float v_length;
varying float v_alpha;
varying vec2 v_texcoord;
varying vec2 v_bevel_distance;

void main()
{
    float distance = v_texcoord.y;

    // Round join (instead of miter)
    // if (v_texcoord.x < 0.0)          { distance = length(v_texcoord); }
    // else if(v_texcoord.x > v_length) { distance = length(v_texcoord - vec2(v_length, 0.0)); }

    float d = abs(distance) - linewidth/2.0 + antialias;

    // Miter limit
    float m = miter_limit*(linewidth/2.0);
    if (v_texcoord.x < 0.0)          { d = max(v_bevel_distance.x-m ,d); }
    else if(v_texcoord.x > v_length) { d = max(v_bevel_distance.y-m ,d); }

    float alpha = 1.0;
    if( d > 0.0 )
    {
        alpha = d/(antialias);
        alpha = exp(-alpha*alpha);
    }
    gl_FragColor = vec4(0, 0, 0, alpha*v_alpha);
} """

geometry_code = """
#version 120
#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_geometry_shader4 : enable

uniform mat4 projection;
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;

varying out float v_length;
varying out float v_alpha;
varying out vec2 v_texcoord;
varying out vec2 v_bevel_distance;

float compute_u(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    // Then  u *= lenght(p1-p0)
    vec2 v = p1 - p0;
    float l = length(v);
    return ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l;
}

float line_distance(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    vec2 v = p1 - p0;
    float l2 = v.x*v.x + v.y*v.y;
    float u = ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l2;

    // h is the prpjection of p on (p0,p1)
    vec2 h = p0 + u*v;

    return length(p-h);
}

void main(void)
{
    // Get the four vertices passed to the shader
    vec2 p0 = gl_PositionIn[0].xy; // start of previous segment
    vec2 p1 = gl_PositionIn[1].xy; // end of previous segment, start of current segment
    vec2 p2 = gl_PositionIn[2].xy; // end of current segment, start of next segment
    vec2 p3 = gl_PositionIn[3].xy; // end of next segment

    // Determine the direction of each of the 3 segments (previous, current, next)
    vec2 v0 = normalize(p1 - p0);
    vec2 v1 = normalize(p2 - p1);
    vec2 v2 = normalize(p3 - p2);

    // Determine the normal of each of the 3 segments (previous, current, next)
    vec2 n0 = vec2(-v0.y, v0.x);
    vec2 n1 = vec2(-v1.y, v1.x);
    vec2 n2 = vec2(-v2.y, v2.x);

    // Determine miter lines by averaging the normals of the 2 segments
    vec2 miter_a = normalize(n0 + n1); // miter at start of current segment
    vec2 miter_b = normalize(n1 + n2); // miter at end of current segment

    // Determine the length of the miter by projecting it onto normal
    vec2 p,v;
    float d;
    float w = linewidth/2.0 + 1.5*antialias;
    v_length = length(p2-p1);

    float length_a = w / dot(miter_a, n1);
    float length_b = w / dot(miter_b, n1);

    float m = miter_limit*linewidth/2.0;

    // Angle between prev and current segment (sign only)
    float d0 = +1.0;
    if( (v0.x*v1.y - v0.y*v1.x) > 0 ) { d0 = -1.0;}

    // Angle between current and next segment (sign only)
    float d1 = +1.0;
    if( (v1.x*v2.y - v1.y*v2.x) > 0 ) { d1 = -1.0; }

    // Generate the triangle strip

    v_alpha = 1.0;
    // Cap at start
    if( p0 == p1 ) {
        p = p1 - w*v1 + w*n1;
        gl_Position = projection*vec4(p, 0.0, 1.0);
        v_texcoord = vec2(-w, +w);
        if (p2 == p3) v_alpha = 0.0;
    // Regular join
    } else {
        p = p1 + length_a * miter_a;
        gl_Position = projection*vec4(p, 0.0, 1.0);
        v_texcoord = vec2(compute_u(p1,p2,p), +w);
    }
    v_bevel_distance.x = +d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    v_alpha = 1.0;
    // Cap at start
    if( p0 == p1 ) {
        p = p1 - w*v1 - w*n1;
        v_texcoord = vec2(-w, -w);
        if (p2 == p3) v_alpha = 0.0;
    // Regular join
    } else {
        p = p1 - length_a * miter_a;
        v_texcoord = vec2(compute_u(p1,p2,p), -w);
    }
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = -d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    v_alpha = 1.0;
    // Cap at end
    if( p2 == p3 ) {
        p = p2 + w*v1 + w*n1;
        v_texcoord = vec2(v_length+w, +w);
        if (p0 == p1) v_alpha = 0.0;
    // Regular join
    } else {
        p = p2 + length_b * miter_b;
        v_texcoord = vec2(compute_u(p1,p2,p), +w);
    }
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = +d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    v_alpha = 1.0;
    // Cap at end
    if( p2 == p3 ) {
        p = p2 + w*v1 - w*n1;
        v_texcoord = vec2(v_length+w, -w);
        if (p0 == p1) v_alpha = 0.0;
    // Regular join
    } else {
        p = p2 - length_b * miter_b;
        v_texcoord = vec2(compute_u(p1,p2,p), -w);
    }
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = -d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    EndPrimitive();
}
"""

# GLUT callbacks
# --------------------------------------
def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(GL_LINE_STRIP_ADJACENCY_EXT, 0, len(data))
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
glut.glutCreateWindow('Antialiased lines using geometry shader')
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
antialias = 1.0
miter_limit = 4.0

# Splitted lines (each line must have at least 3 points)
linewidth = 35.0
data = np.array(
    [(100,300), (100,300), (350,300), (700,300), (700,300),
     (100,500), (100,500), (350,500), (700,500), (700,500)]).astype(np.float32)

# Nice spiral
n = 1024
linewidth = 1.0
T = np.linspace(0, 10*2*np.pi, n)
R = np.linspace(10, 400, n)
data = np.zeros((n,2), dtype=np.float32)
data[:,0] = 400 + np.cos(T)*R
data[:,1] = 400 + np.sin(T)*R

# Star
# n = 12
# miter_limit = 1.0
# linewidth = 20.0
# data = np.zeros((2*n+2,2),dtype=np.float32)
# data[1:-1] = (star(n=12)*400 + (400,400)).astype(np.float32)
# data[0] = data[1]
# data[-1] = data[-2]


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
geometry = gl.glCreateShader(GL_GEOMETRY_SHADER_EXT)
gl.glShaderSource(geometry, geometry_code)
gl.glCompileShader(geometry)
glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_EXT, 4)
glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES_ADJACENCY_EXT)
glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, gl.GL_TRIANGLE_STRIP)
gl.glAttachShader(program, geometry)
gl.glLinkProgram(program)
gl.glDetachShader(program, vertex)
gl.glDetachShader(program, fragment)
gl.glDetachShader(program, geometry)
gl.glUseProgram(program)


# Build buffer & bind attributes
# --------------------------------------
vbuffer = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbuffer)
gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
stride = data.strides[0]
offset = ctypes.c_void_p(0)
loc = gl.glGetAttribLocation(program, "position")
gl.glEnableVertexAttribArray(loc)
gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)
loc = gl.glGetUniformLocation(program, "linewidth")
gl.glUniform1f(loc, linewidth)
loc = gl.glGetUniformLocation(program, "antialias")
gl.glUniform1f(loc, antialias)
loc = gl.glGetUniformLocation(program, "miter_limit")
gl.glUniform1f(loc, miter_limit);

# OpenGL initalization
# --------------------------------------
gl.glClearColor(1,1,1,1)
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

# Enter mainloop
# --------------------------------------
glut.glutMainLoop()
