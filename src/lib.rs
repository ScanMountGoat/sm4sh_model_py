use pyo3::prelude::*;

#[macro_export]
macro_rules! python_enum {
    ($py_ty:ident, $rust_ty:ty, $( $i:ident ),+) => {
        #[pyclass(eq, eq_int)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $py_ty {
            $($i),*
        }

        // These will generate a compile error if variant names don't match.
        impl From<$rust_ty> for $py_ty {
            fn from(value: $rust_ty) -> Self {
                match value {
                    $(<$rust_ty>::$i => Self::$i),*
                }
            }
        }

        impl From<$py_ty> for $rust_ty {
            fn from(value: $py_ty) -> Self {
                match value {
                    $(<$py_ty>::$i => Self::$i),*
                }
            }
        }

        impl ::map_py::MapPy<$rust_ty> for $py_ty {
            fn map_py(self, _py: Python) -> PyResult<$rust_ty> {
                Ok(self.into())
            }
        }

        impl ::map_py::MapPy<$py_ty> for $rust_ty {
            fn map_py(self, _py: Python) -> PyResult<$py_ty> {
                Ok(self.into())
            }
        }
    };
}

// Match the module hierarchy and types of sm4sh_model as closely as possible.
#[pymodule]
mod sm4sh_model_py {
    use super::*;

    #[pymodule]
    mod nud {
        use map_py::{MapPy, TypedList};
        use numpy::PyArray1;
        use pyo3::prelude::*;

        #[pyfunction]
        fn load_model(py: Python, path: &str) -> PyResult<NudModel> {
            let model = sm4sh_model::nud::load_model(path);
            model.map_py(py)
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudModel)]
        pub struct NudModel {
            pub groups: TypedList<NudMeshGroup>,
            pub textures: TypedList<ImageTexture>,
            pub bone_start_index: usize,
            pub bone_end_index: usize,
            pub bounding_sphere: [f32; 4],
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudMeshGroup)]
        pub struct NudMeshGroup {
            pub name: String,
            pub meshes: TypedList<NudMesh>,
            pub sort_bias: f32,
            pub bounding_sphere: [f32; 4],
            pub bone_flags: BoneFlags,
            pub parent_bone_index: Option<usize>,
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudMesh)]
        pub struct NudMesh {
            pub vertices: vertex::Vertices,
            pub vertex_indices: Py<PyArray1<u16>>,
            pub unk3: bool,
            pub primitive_type: PrimitiveType,
            pub material1: Option<NudMaterial>,
            pub material2: Option<NudMaterial>,
            pub material3: Option<NudMaterial>,
            pub material4: Option<NudMaterial>,
        }

        #[pymodule]
        mod vertex {
            use map_py::{MapPy, TypedList};
            use numpy::PyArray2;
            use pyo3::prelude::*;

            #[pyclass(get_all, set_all)]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Vertices)]
            pub struct Vertices {
                pub positions: Py<PyArray2<f32>>,
                pub normals: Normals,
                pub bones: Bones,
                pub colors: Colors,
                pub uvs: TypedList<Uvs>,
            }

            // TODO: Rework these to be representable using numpy arrays.
            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Normals)]
            pub struct Normals(pub sm4sh_model::nud::vertex::Normals);

            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Bones)]
            pub struct Bones(pub sm4sh_model::nud::vertex::Bones);

            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Colors)]
            pub struct Colors(pub sm4sh_model::nud::vertex::Colors);

            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Uvs)]
            pub struct Uvs(pub sm4sh_model::nud::vertex::Uvs);
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudMaterial)]
        pub struct NudMaterial {
            // TODO: Should this recreate flags or store them directly?
            #[map(from(map_py::helpers::into), into(map_py::helpers::try_into))]
            pub flags: u32,
            pub src_factor: SrcFactor,
            pub dst_factor: DstFactor,
            pub alpha_func: AlphaFunc,
            pub cull_mode: CullMode,
            pub textures: TypedList<NudTexture>,
            pub properties: TypedList<NudProperty>,
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudTexture)]
        pub struct NudTexture {
            pub hash: u32,
            pub map_mode: MapMode,
            pub wrap_mode_s: WrapMode,
            pub wrap_mode_t: WrapMode,
            pub min_filter: MinFilter,
            pub mag_filter: MagFilter,
            pub mip_detail: MipDetail,
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudProperty)]
        pub struct NudProperty {
            pub name: String,
            pub values: Vec<f32>,
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::ImageTexture)]
        pub struct ImageTexture {
            pub hash_id: u32,
            pub width: u32,
            pub height: u32,
            pub mipmap_count: u32,
            pub image_format: NutFormat,
            pub image_data: Vec<u8>,
        }

        python_enum!(
            BoneFlags,
            sm4sh_model::nud::BoneFlags,
            Disabled,
            Skinning,
            ParentBone
        );

        python_enum!(
            PrimitiveType,
            sm4sh_model::nud::PrimitiveType,
            TriangleList,
            TriangleStrip
        );

        python_enum!(
            SrcFactor,
            sm4sh_model::nud::SrcFactor,
            One,
            SourceAlpha,
            One2,
            SourceAlpha2,
            Zero,
            SourceAlpha3,
            DestinationAlpha,
            DestinationAlpha7,
            DestinationColor,
            SrcAlpha3,
            SrcAlpha4,
            Unk16,
            Unk33,
            SrcAlpha5
        );

        python_enum!(
            DstFactor,
            sm4sh_model::nud::DstFactor,
            Zero,
            OneMinusSourceAlpha,
            One,
            OneReverseSubtract,
            SourceAlpha,
            SourceAlphaReverseSubtract,
            OneMinusDestinationAlpha,
            One2,
            Zero2,
            Unk10,
            OneMinusSourceAlpha2,
            One3,
            Zero5,
            Zero3,
            One4,
            OneMinusSourceAlpha3,
            One5
        );

        python_enum!(
            AlphaFunc,
            sm4sh_model::nud::AlphaFunc,
            Disabled,
            Never,
            Less,
            Eq,
            Leq,
            Neq,
            Geq,
            Always
        );

        python_enum!(
            CullMode,
            sm4sh_model::nud::CullMode,
            Disabled,
            Outside,
            Inside,
            Disabled2,
            Inside2,
            Outside2
        );

        python_enum!(
            MapMode,
            sm4sh_model::nud::MapMode,
            TexCoord,
            EnvCamera,
            Projection,
            EnvLight,
            EnvSpec
        );

        python_enum!(
            MinFilter,
            sm4sh_model::nud::MinFilter,
            LinearMipmapLinear,
            Nearest,
            Linear,
            NearestMipmapLinear
        );

        python_enum!(
            MagFilter,
            sm4sh_model::nud::MagFilter,
            Unk0,
            Nearest,
            Linear
        );

        python_enum!(
            MipDetail,
            sm4sh_model::nud::MipDetail,
            OneMipLevelAnisotropicOff,
            Unk1,
            OneMipLevelAnisotropicOff2,
            FourMipLevels,
            FourMipLevelsAnisotropic,
            FourMipLevelsTrilinear,
            FourMipLevelsTrilinearAnisotropic
        );

        python_enum!(
            WrapMode,
            sm4sh_model::nud::WrapMode,
            Repeat,
            MirroredRepeat,
            ClampToEdge
        );

        python_enum!(
            NutFormat,
            sm4sh_model::nud::NutFormat,
            BC1Unorm,
            BC2Unorm,
            BC3Unorm,
            Bgr5A1Unorm,
            Bgr5A1Unorm2,
            B5G6R5Unorm,
            Rgb5A1Unorm,
            Rgba8Unorm,
            R32Float,
            Rgba82,
            BC5Unorm
        );
    }
}
