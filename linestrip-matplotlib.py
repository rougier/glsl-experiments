#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (C) 2013 Nicolas P. Rougier. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY NICOLAS P. ROUGIER ''AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL NICOLAS P. ROUGIER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Nicolas P. Rougier.
# ----------------------------------------------------------------------------
import math
import matplotlib
import numpy as np
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Arc
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects

# -----------------------------------------------------------------------------
def figure(width=800, height=800, on_key=None):
    """ """

    dpi = 72.0
    figsize= width/float(dpi), height/float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    if on_key:
        fig.canvas.mpl_connect('key_press_event', on_key)
    axes = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    axes.set_xlim(0, width)
    axes.set_ylim(0, height)
    plt.xticks([])
    plt.yticks([])

# -----------------------------------------------------------------------------
def line(verts, color = 'k', linewidth=1, alpha=1, capstyle='butt', joinstyle='miter'):

    verts = np.array(verts).reshape(len(verts),2)
    codes = [Path.MOVETO] + [Path.LINETO,]*(len(verts)-1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path,
                              linewidth = linewidth,
                              edgecolor = color,
                              facecolor = 'None',
                              alpha     = alpha)
    patch.set_path_effects([PathEffects.Stroke(capstyle=capstyle,
                                               joinstyle=joinstyle )])
    plt.gca().add_patch(patch)
    plt.xticks([]), plt.yticks([])


def arrow(origin, direction, color = 'k', linewidth=2, alpha=1):
    x0, y0 = origin
    dx, dy = direction
    plt.arrow(x0, y0, dx, dy, head_width=5, head_length=5,
              fc=color, ec=color, lw=linewidth)

def disc(verts, color = 'k', linewidth=1, alpha=1, edgecolor='k', facecolor='w'):
    verts = np.array(verts).reshape(len(verts),2)
    plt.scatter(verts[:,0], verts[:,1], linewidth=linewidth,
                s=25, edgecolor=edgecolor, facecolor=facecolor, zorder=10)

# -----------------------------------------------------------------------------
def normalize(P):
    return P/np.sqrt((P*P).sum())

def length(P):
    return np.sqrt((P*P).sum())

def tangent(A,B):
    T = B - A
    return T/np.sqrt((T*T).sum())

def ortho(A,B):
    T = tangent(A,B)
    return np.array([-T[1],T[0]])

def dot(A,B):
    return np.dot(A,B)

def cross(A,B):
    return np.cross(A,B)

def compute_u(p0, p1, p):
    # Projection p' of p such that p' = p0 + u*(p1-p0)
    # Then  u *= lenght(p1-p0)
    v = p1 - p0
    l = length(v)
    return ((p[0]-p0[0])*v[0] + (p[1]-p0[1])*v[1]) / l

def line_distance(p0, p1, p):
    # Projection p' of p such that p' = p0 + u*(p1-p0)
    v = p1 - p0
    l2 = v[0]*v[0] + v[1]*v[1]
    u = ((p[0]-p0[0])*v[0] + (p[1]-p0[1])*v[1]) / l2
    # h is the projection of p on (p0,p1)
    h = p0 + u*v;
    return length(p-h)

def geometry_shader(p0,p1,p2,p3,linewidth,antialias,miter_limit):
    # Determine the direction of each of the 3 segments (previous, current, next)
    v0 = normalize(p1 - p0);
    v1 = normalize(p2 - p1);
    v2 = normalize(p3 - p2);

    # Determine the normal of each of the 3 segments (previous, current, next)
    n0 = np.array([-v0[1], v0[0]])
    n1 = np.array([-v1[1], v1[0]])
    n2 = np.array([-v2[1], v2[0]])

    # Determine miter lines by averaging the normals of the 2 segments
    miter_a = normalize(n0 + n1) # miter at start of current segment
    miter_b = normalize(n1 + n2) # miter at end of current segment

    # Determine the length of the miter by projecting it onto normal
    w = linewidth/2.0 + 1.0 + antialias
    l = length(p2-p1)

    length_a = w / dot(miter_a, n1)
    length_b = w / dot(miter_b, n1)

#    length_a = min(l,length_a)
#    length_b = min(l,length_b)

    m = miter_limit*linewidth/2.0;

    # // Angle between prev and current segment (sign only)
    # float d0 = +1.0;
    # if( (v0.x*v1.y - v0.y*v1.x) > 0 ) { d0 = -1.0;}

    # // Angle between current and next segment (sign only)
    # float d1 = +1.0;
    # if( (v1.x*v2.y - v1.y*v2.x) > 0 ) { d1 = -1.0; }

    # // Generate the triangle strip
    # // ---------------------------
    A = p1 + length_a * miter_a;
    # gl_Position = projection*vec4(p, 0.0, 1.0);
    # v_texcoord = vec2(compute_u(p1,p2,p), +w);
    # v_bevel_distance.x = +d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    # v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);

    B = p1 - length_a * miter_a;
    # gl_Position = projection*vec4(p, 0.0, 1.0);
    # v_texcoord = vec2(compute_u(p1,p2,p), -w);
    # v_bevel_distance.x = -d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    # v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);

    C = p2 + length_b * miter_b;
    # gl_Position = projection*vec4(p, 0.0, 1.0);
    # v_texcoord = vec2(compute_u(p1,p2,p), +w);
    # v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    # v_bevel_distance.y = +d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);

    D = p2 - length_b * miter_b;
    # gl_Position = projection*vec4(p, 0.0, 1.0);
    # v_texcoord = vec2(compute_u(p1,p2,p), -w);
    # v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    # v_bevel_distance.y = -d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);

    return np.array([A,B,C,D]).reshape(4,2)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    fig = figure()

    antialias= 1.0
    miter_limit = 4.0
    linewidth = 95.0
    d = 50
    h=-50
    data = [ (0,       256.5),
             (28,     256.5),
             (256-d,   256.5),
             (256,     256.5+128),
             (256+d,   256.5+h),
             (512-28, 256.5+h),
             (512,     256.5+h) ]
    data = np.array(data).astype(np.float32)
    data += (200,100)

    for i in range(len(data)-3):
        p0 = data[i+0]
        p1 = data[i+1]
        p2 = data[i+2]
        p3 = data[i+3]
        V = geometry_shader(p0,p1,p2,p3,antialias,linewidth,miter_limit)
        disc(V)
        line([V[0],V[1],V[3],V[2],V[0]], color='.5')

    line(data[1:-1], linewidth=2)
    plt.show()

    # linewidth = 100

    # P0 = np.array((200,400))
    # P1 = np.array((400,300))
    # P2 = np.array((600,400))

    # line((P0,P1,P2), linewidth=linewidth, alpha=.1, capstyle='projecting')
    # line((P0,P1), linewidth=linewidth, alpha=.25)
    # line((P0,P1), linewidth=1)
    # line((P1,P2), linewidth=linewidth, alpha=.25)
    # line((P1,P2), linewidth=1)

    # T1 = tangent(P0,P1)
    # O1 = ortho(P0,P1)
    # arrow( P0, (linewidth/4)*T1 )
    # arrow( P0, (linewidth/4)*O1 )

    # T2 = tangent(P1,P2)
    # O2 = ortho(P1,P2)
    # arrow( P1, (linewidth/4)*T2 )
    # arrow( P1, (linewidth/4)*O2 )


    # A1 = P0 + O1 * 0.5*linewidth
    # A2 = P0 - O1 * 0.5*linewidth
    # D1 = P2 + O2 * 0.5*linewidth
    # D2 = P2 - O2 * 0.5*linewidth
    # disc([A1,A2,D1,D2])

    # B1 = P1 + O1 * 0.5*linewidth
    # B2 = P1 - O1 * 0.5*linewidth
    # C1 = P1 + O2 * 0.5*linewidth
    # C2 = P1 - O2 * 0.5*linewidth
    # if np.cross(P1-P0,P2-P1) > 0:
    #     disc([B2,C2], edgecolor='r', facecolor='w')
    #     disc([B1,C1])
    # else:
    #     disc([B1,C1], edgecolor='r', facecolor='w')
    #     disc([B2,C2])

    # O = normalize(O1+O2)
    # M1 = P1 + O * 0.5*linewidth / dot(O,O1)
    # M2 = P1 - O * 0.5*linewidth / dot(O,O1)
    # disc([M1,M2])

    # # arrow( P1, +O * 0.5*linewidth / dot(O,O1) )
    # # arrow( P1, -O * 0.5*linewidth / dot(O,O1) )

    plt.show()
