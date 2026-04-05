precision highp float;

uniform sampler2D texture;

// Passed from the vertex shader
varying vec2 fragTextureCoord;
varying float fragAlpha;

// float linearstep(float edge0, float edge1, float x) {
//   return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
// }

void main() {
  // Test if the fragment is inside the circle area of this point
  // vec2 cxy = 2.0 * gl_PointCoord - 1.0;
  // if (dot(cxy, cxy) > 1.03) {
  //   discard;
  //   return;
  // }

  // Need to pre-compute alpha
  // https://medium.com/david-guan/alpha-blending-and-webgl-823d86de00d8
  vec4 color = texture2D(texture, fragTextureCoord);
  vec4 colorOpacity = color * fragAlpha * color[3];
  gl_FragColor = colorOpacity;
}