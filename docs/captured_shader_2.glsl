#version 300 es
#define attribute in
#define varying out
#define texture2D texture
precision highp float;
	precision highp int;
	precision highp sampler2D;
	precision highp samplerCube;
	precision highp sampler3D;
	precision highp sampler2DArray;
	precision highp sampler2DShadow;
	precision highp samplerCubeShadow;
	precision highp sampler2DArrayShadow;
	precision highp isampler2D;
	precision highp isampler3D;
	precision highp isamplerCube;
	precision highp isampler2DArray;
	precision highp usampler2D;
	precision highp usampler3D;
	precision highp usamplerCube;
	precision highp usampler2DArray;

#define HIGH_PRECISION
#define SHADER_TYPE ShaderMaterial
#define SHADER_NAME
#define DOUBLE_SIDED
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
uniform bool isOrthographic;
#ifdef USE_INSTANCING
	attribute mat4 instanceMatrix;
#endif
#ifdef USE_INSTANCING_COLOR
	attribute vec3 instanceColor;
#endif
#ifdef USE_INSTANCING_MORPH
	uniform sampler2D morphTexture;
#endif
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
#ifdef USE_UV1
	attribute vec2 uv1;
#endif
#ifdef USE_UV2
	attribute vec2 uv2;
#endif
#ifdef USE_UV3
	attribute vec2 uv3;
#endif
#ifdef USE_TANGENT
	attribute vec4 tangent;
#endif
#if defined( USE_COLOR_ALPHA )
	attribute vec4 color;
#elif defined( USE_COLOR )
	attribute vec3 color;
#endif
#ifdef USE_SKINNING
	attribute vec4 skinIndex;
	attribute vec4 skinWeight;
#endif
#define USE_SKINNING
        precision highp float;
#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].y );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated
        attribute uint splatIndex;
        uniform highp usampler2D flameModelTexture;
        uniform highp usampler2D boneTexture;
        uniform highp usampler2D boneWeightTexture;
        uniform highp usampler2D centersColorsTexture;
        uniform highp sampler2D sphericalHarmonicsTexture;
        uniform highp sampler2D sphericalHarmonicsTextureR;
        uniform highp sampler2D sphericalHarmonicsTextureG;
        uniform highp sampler2D sphericalHarmonicsTextureB;
        uniform highp usampler2D sceneIndexesTexture;
        uniform vec2 sceneIndexesTextureSize;
        uniform int sceneCount;
        uniform int gaussianSplatCount;
        uniform int bsCount;
        uniform float headBoneIndex;
        #ifdef USE_SKINNING
            attribute vec4 skinIndex;
            attribute vec4 skinWeight;
        #endif


            uniform vec2 covariancesTextureSize;
            uniform highp sampler2D covariancesTexture;
            uniform highp usampler2D covariancesTextureHalfFloat;
            uniform int covariancesAreHalfFloat;
            void fromCovarianceHalfFloatV4(uvec4 val, out vec4 first, out vec4 second) {
                vec2 r = unpackHalf2x16(val.r);
                vec2 g = unpackHalf2x16(val.g);
                vec2 b = unpackHalf2x16(val.b);
                first = vec4(r.x, r.y, g.x, g.y);
                second = vec4(b.x, b.y, 0.0, 0.0);
            }

        uniform vec2 focal;
        uniform float orthoZoom;
        uniform int orthographicMode;
        uniform int pointCloudModeEnabled;
        uniform float inverseFocalAdjustment;
        uniform vec2 viewport;
        uniform vec2 basisViewport;
        uniform vec2 centersColorsTextureSize;
        uniform vec2 flameModelTextureSize;
        uniform vec2 boneWeightTextureSize;
        uniform vec2 boneTextureSize;
        uniform int sphericalHarmonicsDegree;
        uniform vec2 sphericalHarmonicsTextureSize;
        uniform int sphericalHarmonics8BitMode;
        uniform int sphericalHarmonicsMultiTextureMode;
        uniform float visibleRegionRadius;
        uniform float visibleRegionFadeStartRadius;
        uniform float firstRenderTime;
        uniform float currentTime;
        uniform int fadeInComplete;
        uniform vec3 sceneCenter;
        uniform float splatScale;
        uniform float sphericalHarmonics8BitCompressionRangeMin[32];
        uniform float sphericalHarmonics8BitCompressionRangeMax[32];
        varying vec4 vColor;
        varying vec2 vUv;
        varying vec2 vPosition;
        varying vec2 vSplatIndex;
        #ifdef USE_SKINNING
            uniform mat4 bindMatrix;
            uniform mat4 bindMatrixInverse;
            uniform highp sampler2D boneTexture0;
            mat4 getBoneMatrix0( const in float i ) {
                int size = textureSize( boneTexture0, 0 ).x;
                int j = int( i ) * 4;
                int x = j % size;
                int y = j / size;
                vec4 v1 = texelFetch( boneTexture0, ivec2( x, y ), 0 );
                vec4 v2 = texelFetch( boneTexture0, ivec2( x + 1, y ), 0 );
                vec4 v3 = texelFetch( boneTexture0, ivec2( x + 2, y ), 0 );
                vec4 v4 = texelFetch( boneTexture0, ivec2( x + 3, y ), 0 );
                return mat4( v1, v2, v3, v4 );
            }
        #endif
        mat3 quaternionToRotationMatrix(float x, float y, float z, float w) {
            float s = 1.0 / sqrt(w * w + x * x + y * y + z * z);

            return mat3(
                1. - 2. * (y * y + z * z),
                2. * (x * y + w * z),
                2. * (x * z - w * y),
                2. * (x * y - w * z),
                1. - 2. * (x * x + z * z),
                2. * (y * z + w * x),
                2. * (x * z + w * y),
                2. * (y * z - w * x),
                1. - 2. * (x * x + y * y)
            );
        }
        const float sqrt8 = sqrt(8.0);
        const float minAlpha = 1.0 / 255.0;
        const vec4 encodeNorm4 = vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
        const uvec4 mask4 = uvec4(uint(0x000000FF), uint(0x0000FF00), uint(0x00FF0000), uint(0xFF000000));
        const uvec4 shift4 = uvec4(0, 8, 16, 24);
        int internal = 1;//show a gaussian splatting point every internal points.
        vec4 uintToRGBAVec (uint u) {
           uvec4 urgba = mask4 & u;
           urgba = urgba >> shift4;
           vec4 rgba = vec4(urgba) * encodeNorm4;
           return rgba;
        }
        float getRealIndex(int sIndex, int reducedFactor) {
            int remainder = sIndex % reducedFactor;
            if(remainder == int(0)) {
                return float(sIndex);
            }
            else
            {
                return float(sIndex - remainder);
            }
        }
        vec2 getDataUV(in int stride, in int offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(getRealIndex(int(splatIndex), internal)) * uint(stride) + uint(offset)) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }
        vec2 getFlameDataUV(in int stride, in int offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(int(splatIndex) / internal) * uint(stride) + uint(offset) + uint(gaussianSplatCount * bsCount)) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }
        vec2 getBoneWeightUV(in int stride, in int offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(int(splatIndex) / internal) * uint(stride) + uint(offset)) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }
        vec2 getBSFlameDataUV(in int bsInedex, in int stride, in int offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(int(splatIndex) / internal) * uint(stride) + uint(offset) + uint(gaussianSplatCount * bsInedex)) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }
        vec2 getDataUVF(in uint sIndex, in float stride, in uint offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(float(getRealIndex(int(sIndex), internal)) * stride) + offset) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }
        const float SH_C1 = 0.4886025119029199f;
        const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);
        mat4 getBoneMatrix( float i ) {
            float y = i;
            float x = 0.0;
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(i * 4.0) / boneTextureSize.x;//4
            samplerUV.y = float(floor(d)) / boneTextureSize.y;//5
            samplerUV.x = fract(d);
            vec4 v1 = uintBitsToFloat(texture( boneTexture, samplerUV ));
            vec4 v2 = uintBitsToFloat(texture( boneTexture, vec2(samplerUV.x + 1.0 / boneTextureSize.x, samplerUV.y)));
            vec4 v3 = uintBitsToFloat(texture( boneTexture, vec2(samplerUV.x + 2.0 / boneTextureSize.x, samplerUV.y) ));
            vec4 v4 = uintBitsToFloat(texture( boneTexture, vec2(samplerUV.x + 3.0 / boneTextureSize.x, samplerUV.y)));
            return mat4( v1, v2, v3, v4 );
        }
        void main () {
            uint oddOffset = splatIndex & uint(0x00000001);
            uint doubleOddOffset = oddOffset * uint(2);
            bool isEven = oddOffset == uint(0);
            uint nearestEvenIndex = splatIndex - oddOffset;
            float fOddOffset = float(oddOffset);
            uvec4 sampledCenterColor = texture(centersColorsTexture, getDataUV(1, 0, centersColorsTextureSize));
            // vec3 splatCenter = uintBitsToFloat(uvec3(sampledCenterColor.gba));
            uvec3 sampledCenter = texture(centersColorsTexture, getDataUV(1, 0, centersColorsTextureSize)).gba;
            vec3 splatCenter = uintBitsToFloat(uvec3(sampledCenter));
            vec2 flameTextureUV = getBSFlameDataUV(bsCount, 1, 0, flameModelTextureSize);
            uvec3 sampledflamePos = texture(flameModelTexture, flameTextureUV).rgb;
            // splatCenter += uintBitsToFloat(uvec3(sampledflamePos.rgb));
            for(int i = 0; i < bsCount; ++i) {
                vec2 flameBSTextureUV = getBSFlameDataUV(i, 1, 0, flameModelTextureSize);
                uvec3 sampledBSPos = texture(flameModelTexture, flameBSTextureUV).rgb;
                vec2 samplerUV = vec2(0.0, 0.0);
                float d = float(i / 4 + 5 * 4) / boneTextureSize.x;//4
                samplerUV.y = float(floor(d)) / boneTextureSize.y;//32
                samplerUV.x = fract(d);
                vec4 bsWeight = uintBitsToFloat(texture(boneTexture, samplerUV));
                float weight = bsWeight.r;
                if(i % 4 == 1) {
                    weight = bsWeight.g;
                }
                if(i % 4 == 2) {
                    weight = bsWeight.b;
                }
                if(i % 4 == 3) {
                    weight = bsWeight.a;
                }
                splatCenter = splatCenter + weight * uintBitsToFloat(sampledBSPos);
            }
            #ifdef USE_SKINNING
                mat4 boneMatX = getBoneMatrix0( skinIndex.x );
                mat4 boneMatY = getBoneMatrix0( skinIndex.y );
                mat4 boneMatZ = getBoneMatrix0( skinIndex.z );
                mat4 boneMatW = getBoneMatrix0( skinIndex.w );
            #endif
            #ifdef USE_SKINNING
                mat4 skinMatrix = mat4( 0.0 );
                skinMatrix += skinWeight.x * boneMatX;
                skinMatrix += skinWeight.y * boneMatY;
                skinMatrix += skinWeight.z * boneMatZ;
                skinMatrix += skinWeight.w * boneMatW;
                // skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
            #endif
            vec3 transformed = vec3(splatCenter.xyz);
            #ifdef USE_SKINNING
                // vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
                vec4 skinVertex = vec4( transformed, 1.0 );
                vec4 skinned = vec4( 0.0 );
                // There is an offset between the Gaussian point and the mesh vertex,
                // which will cause defects in the skeletal animation driving the Gaussian point.
                //In order to circumvent this problem, only the head bone(index is 110 currently) is used to drive
                if (headBoneIndex >= 0.0)
                {
                    mat4 boneMat = getBoneMatrix0( headBoneIndex );
                    skinned += boneMat * skinVertex * 1.0;
                }
                // skinned += boneMatX * skinVertex * skinWeight.x;
                // skinned += boneMatY * skinVertex * skinWeight.y;
                // skinned += boneMatZ * skinVertex * skinWeight.z;
                // skinned += boneMatW * skinVertex * skinWeight.w;
                // transformed = ( bindMatrixInverse * skinned ).xyz;
                transformed = skinned.xyz;
            #endif
            splatCenter = transformed.xyz;
            #ifdef USE_FLAME
                mat4 boneMatX = getBoneMatrix( 0.0 );
                mat4 boneMatY = getBoneMatrix( 1.0 );
                mat4 boneMatZ = getBoneMatrix( 2.0 );
                mat4 boneMatW = getBoneMatrix( 3.0 );
                mat4 boneMat0 = getBoneMatrix( 4.0 );

                vec2 boneWeightUV0 = getBoneWeightUV(2, 0, boneWeightTextureSize);
                vec2 boneWeightUV1 = getBoneWeightUV(2, 1, boneWeightTextureSize);
                uvec4 sampledBoneMatrixValue = texture(boneWeightTexture, boneWeightUV0);
                uvec4 sampledBoneMatrixValue0 = texture(boneWeightTexture, boneWeightUV1);
                vec4 boneMatrixValue = uintBitsToFloat(sampledBoneMatrixValue);
                vec4 boneMatrixValue0 = uintBitsToFloat(sampledBoneMatrixValue0);
                vec4 skinVertex = vec4( splatCenter, 1.0 );
                vec4 skinned = vec4( 0.0 );
                float minWeight = min(boneMatrixValue.x,min(boneMatrixValue.y, min(boneMatrixValue.z, min(boneMatrixValue.w, boneMatrixValue0.x))));

                if(boneMatrixValue.x > 0.0 && boneMatrixValue.x > minWeight)
                    skinned += boneMatX * skinVertex * boneMatrixValue.x;

                if(boneMatrixValue.y > 0.0 && boneMatrixValue.y > minWeight)
                    skinned += boneMatY * skinVertex * boneMatrixValue.y;

                if(boneMatrixValue.z > 0.0 && boneMatrixValue.z > minWeight)
                    skinned += boneMatZ * skinVertex * boneMatrixValue.z;

                if(boneMatrixValue.w > 0.0 && boneMatrixValue.w > minWeight)
                    skinned += boneMatW * skinVertex * boneMatrixValue.w;

                if(boneMatrixValue0.x > 0.0 && boneMatrixValue0.x > minWeight)
                    skinned += boneMat0 * skinVertex * boneMatrixValue0.x;

                splatCenter = skinned.xyz;
            #endif
            uint sceneIndex = uint(0);
            if (sceneCount > 1) {
                sceneIndex = texture(sceneIndexesTexture, getDataUV(1, 0, sceneIndexesTextureSize)).r;
            }
            mat4 transformModelViewMatrix = modelViewMatrix;
            float sh8BitCompressionRangeMinForScene = sphericalHarmonics8BitCompressionRangeMin[sceneIndex];
            float sh8BitCompressionRangeMaxForScene = sphericalHarmonics8BitCompressionRangeMax[sceneIndex];
            float sh8BitCompressionRangeForScene = sh8BitCompressionRangeMaxForScene - sh8BitCompressionRangeMinForScene;
            float sh8BitCompressionHalfRangeForScene = sh8BitCompressionRangeForScene / 2.0;
            vec3 vec8BitSHShift = vec3(sh8BitCompressionRangeMinForScene);
            vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);
            vec4 clipCenter = projectionMatrix * viewCenter;
            float clip = 1.2 * clipCenter.w;
            if (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                return;
            }
            vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
            vPosition = position.xy;
            vSplatIndex = vec2(splatIndex, splatIndex);
            vColor = uintToRGBAVec(sampledCenterColor.r);

            vec4 sampledCovarianceA;
            vec4 sampledCovarianceB;
            vec3 cov3D_M11_M12_M13;
            vec3 cov3D_M22_M23_M33;
            if (covariancesAreHalfFloat == 0) {
                sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset,
                                                                            covariancesTextureSize));
                sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1),
                                                                            covariancesTextureSize));
                cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceB.gba) * fOddOffset;
            } else {
                uvec4 sampledCovarianceU = texture(covariancesTextureHalfFloat, getDataUV(1, 0, covariancesTextureSize));
                fromCovarianceHalfFloatV4(sampledCovarianceU, sampledCovarianceA, sampledCovarianceB);
                cov3D_M11_M12_M13 = sampledCovarianceA.rgb;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg);
            }

            // Construct the 3D covariance matrix
            mat3 Vrk = mat3(
                cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
                cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
                cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
            );
            mat3 J;
            if (orthographicMode == 1) {
                // Since the projection is linear, we don't need an approximation
                J = transpose(mat3(orthoZoom, 0.0, 0.0,
                                0.0, orthoZoom, 0.0,
                                0.0, 0.0, 0.0));
            } else {
                // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
                // 3D covariance matrix instead of using the actual projection matrix because that transformation would
                // require a non-linear component (perspective division) which would yield a non-gaussian result.
                float s = 1.0 / (viewCenter.z * viewCenter.z);
                J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
                    0., 0., 0.
                );
            }
            // Concatenate the projection approximation with the model-view transformation
            mat3 W = transpose(mat3(transformModelViewMatrix));
            mat3 T = W * J;
            // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
            mat3 cov2Dm = transpose(T) * Vrk * T;

                cov2Dm[0][0] += 0.3;
                cov2Dm[1][1] += 0.3;

            // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
            // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
            // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
            // need cov2Dm[1][0] because it is a symetric matrix.
            vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);
            // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
            // so that we can determine the 2D basis for the splat. This is done using the method described
            // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
            // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
            // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * sqrt(eigen-value)), which is
            // equal to scaling them by sqrt(8) standard deviations.
            //
            // This is a different approach than in the original work at INRIA. In that work they compute the
            // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
            // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
            // times the square root of the maximum eigen-value, or 3 standard deviations. They then use the inverse
            // 2D covariance matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by
            // calculating the full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
            float a = cov2Dv.x;
            float d = cov2Dv.z;
            float b = cov2Dv.y;
            float D = a * d - b * b;
            float trace = a + d;
            float traceOver2 = 0.5 * trace;
            float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
            float eigenValue1 = traceOver2 + term2;
            float eigenValue2 = traceOver2 - term2;
            if (pointCloudModeEnabled == 1) {
                eigenValue1 = eigenValue2 = 0.2;
            }
            if (eigenValue2 <= 0.0) return;
            vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
            // since the eigen vectors are orthogonal, we derive the second one from the first
            vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);
            // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
            vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), 1024.0);
            vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), 1024.0);

            vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
                             basisViewport * 2.0 * inverseFocalAdjustment;
            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;
            // Scale the position data we send to the fragment shader
            vPosition *= sqrt8;

            if (fadeInComplete == 0) {
                float opacityAdjust = 1.0;
                float centerDist = length(splatCenter - sceneCenter);
                float renderTime = max(currentTime - firstRenderTime, 0.0);
                float fadeDistance = 0.75;
                float distanceLoadFadeInFactor = step(visibleRegionFadeStartRadius, centerDist);
                distanceLoadFadeInFactor = (1.0 - distanceLoadFadeInFactor) +
                                        (1.0 - clamp((centerDist - visibleRegionFadeStartRadius) / fadeDistance, 0.0, 1.0)) *
                                        distanceLoadFadeInFactor;
                opacityAdjust *= distanceLoadFadeInFactor;
                vColor.a *= opacityAdjust;
            }
        }
