// trying to factor frontend variants into orthogonal parameters

type BoxT = "box_positive" | "box_binary" | "box_textual"
// used eg for box color and for controls needed for box labels
// binary means there are negative and positive boxes
// textual means there is also text associated with the boxes

type LocT =  "loc_coarse" | "loc_fine"
// used for box drawing/ selection controls and for activation display controls

// coarse means boxes span the full image (there are no localized boxes) also only one box at most per im
// fine means can be drawn within image
// Also relevant to activations: fine implies scores/activation are associated with a part of the image

type FrontendT = { box_type : BoxT, loc_type : LocT };

const frontends   = {
        'pytorch': { box_type: "box_positive", loc_type : "loc_fine"} as FrontendT, 
        'default': { box_type: "box_positive", loc_type: "loc_coarse"} as FrontendT,
        // 'multi': {box_type: "box_positive", loc_type : "loc_coarse"}
    }
