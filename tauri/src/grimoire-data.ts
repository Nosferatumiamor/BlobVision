// The Warlock's Grimoire: a built-in catalogue of cryptic SD1.x-era prompt
// fragments (2022-2023 NightCafe/SD1.5-style incantations), browsable and
// insertable from the Prompt/Negative Prompt fields. Content transcribed
// verbatim from BlobVision_Catalogue_Historique_Prompts_SD1x_Base.docx — this
// is a deliberately small seed list the doc itself flags for future growth
// (to 1000-3000 entries pulled from Lexica/DiffusionDB/CivitAI/etc.), not an
// attempt at a complete historical catalogue.

export interface GrimoireCategory {
  name: string;
  entries: string[];
}

export const GRIMOIRE_PROMPT_CATEGORIES: GrimoireCategory[] = [
  {
    name: "Quality",
    entries: [
      "masterpiece",
      "best quality",
      "high quality",
      "ultra quality",
      "masterwork",
      "absurdres",
      "highres",
      "8k",
      "16k",
      "HDR",
      "ultra detailed",
      "highly detailed",
      "intricate",
      "sharp focus",
      "professional",
      "award winning",
    ],
  },
  {
    name: "Websites",
    entries: [
      "trending on artstation",
      "featured on artstation",
      "ArtStation",
      "CGSociety",
      "Pixiv",
      "DeviantArt",
      "Behance",
      "Dribbble",
    ],
  },
  {
    name: "Artists",
    entries: [
      "Greg Rutkowski",
      "Artgerm",
      "WLOP",
      "Guweiz",
      "Alphonse Mucha",
      "Craig Mullins",
      "Ross Tran",
      "Loish",
      "Makoto Shinkai",
      "Moebius",
      "Syd Mead",
      "Beeple",
      "Thomas Kinkade",
      "Ilya Kuvshinov",
      "Charlie Bowater",
      "Peter Mohrbacher",
    ],
  },
  {
    name: "Rendering",
    entries: [
      "octane render",
      "unreal engine",
      "cinema4d",
      "blender",
      "vray",
      "ray tracing",
      "3D render",
      "CG render",
    ],
  },
  {
    name: "Photography",
    entries: [
      "RAW photo",
      "DSLR",
      "Canon EOS R5",
      "Canon 5D",
      "Sony A7R",
      "Nikon D850",
      "Leica M6",
      "35mm",
      "medium format",
      "IMAX",
      "anamorphic lens",
    ],
  },
  {
    name: "Lenses",
    entries: [
      "24mm",
      "35mm lens",
      "50mm",
      "85mm",
      "135mm",
      "macro lens",
      "telephoto",
      "wide angle",
      "f/1.2",
      "f/1.4",
      "f/1.8",
      "f/2.8",
      "shallow depth of field",
      "bokeh",
    ],
  },
  {
    name: "Film Stocks",
    entries: [
      "Kodak Portra",
      "Kodak Gold",
      "Kodachrome",
      "Kodak Vision3",
      "Cinestill 800T",
      "Fujifilm Velvia",
      "Fujifilm Pro 400H",
      "Ilford HP5",
      "Ektachrome",
      "Super 8",
      "GoPro",
      "Polaroid",
    ],
  },
  {
    name: "Lighting",
    entries: [
      "golden hour",
      "blue hour",
      "volumetric lighting",
      "god rays",
      "rim lighting",
      "backlighting",
      "studio lighting",
      "soft lighting",
      "dramatic lighting",
      "ambient light",
      "cinematic lighting",
    ],
  },
  {
    name: "Composition",
    entries: [
      "cinematic",
      "film still",
      "movie still",
      "wide shot",
      "close up",
      "portrait shot",
      "Dutch angle",
      "dynamic pose",
      "full body",
      "upper body",
      "cowboy shot",
      "looking at viewer",
    ],
  },
  {
    name: "Subject Tags",
    entries: ["1girl", "1boy", "1woman", "1man", "solo"],
  },
];

export const GRIMOIRE_NEGATIVE_CATEGORIES: GrimoireCategory[] = [
  {
    name: "Quality",
    entries: ["worst quality", "low quality", "lowres", "blurry", "jpeg artifacts", "pixelated", "oversaturated"],
  },
  {
    name: "Anatomy",
    entries: [
      "bad anatomy",
      "bad hands",
      "bad face",
      "bad proportions",
      "deformed",
      "mutated",
      "mutation",
      "extra limbs",
      "extra arms",
      "extra legs",
      "extra fingers",
      "too many fingers",
      "missing fingers",
      "long neck",
      "cross-eyed",
    ],
  },
  {
    name: "Composition",
    entries: ["cropped", "out of frame", "duplicate", "cloned face", "poorly drawn face", "poorly drawn hands"],
  },
  {
    name: "Artifacts",
    entries: ["text", "watermark", "signature", "logo", "username", "frame", "border"],
  },
];
