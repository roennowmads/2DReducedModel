Shader "Unlit/PointShader"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		ZWrite Off
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"
			#pragma target es3.1

			struct Particle {
				float3 position;
				float3 velocity;
			};

			StructuredBuffer<Particle> _ParticleData;
			//StructuredBuffer<float> _Error;

			Texture2D<float2> _MainTex;

			SamplerState sampler_MainTex
			{
				Filter = MIN_MAG_MIP_POINT;
				AddressU = Wrap;
				AddressV = Wrap;
			};

			uniform matrix model;
			uniform float4 trans;
			uniform float aspect;
			uniform int _PointsCount;
			uniform uint _FrameTime;
			uniform uint _Magnitude;
			uniform int _TextureSwitchFrameNumber;

			struct appdata
			{
				uint id : SV_VertexID;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				fixed3 color : COLOR;
			};

			
			v2f vert (appdata v)
			{
				v2f o;

				uint quadId = v.id;
				float4 particlePos = -float4(_ParticleData[quadId].position, 1.0);
				//particlePos.z += 20.0;

				o.vertex = mul(model, particlePos);
				o.vertex += trans;
				o.vertex = UnityWorldToClipPos(o.vertex.xyz);

				float id = v.id;
				float4 texCoords = float4(id % 64, id / 64, 0.0, 0.0);

				float2 value = _MainTex.SampleLevel(sampler_MainTex, texCoords, 0).rg;
				//float2 value = tex2Dlod(_MainTex, texCoords).rg;

				//float speed = length(_Velocities[quadId]);
				o.color = fixed3(value*10.0, 1.0);//fixed3((normalize(_ParticleData[quadId].velocity) + 1.0) * 0.5);

				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = fixed4(i.color, 1.0);
				return col;
			}
			ENDCG
		}
	}
}
