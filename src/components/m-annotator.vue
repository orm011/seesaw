<template>
  <div
    class="annotator_div row"
    ref="container"
  >
    <img
      :class="read_only ? 'annotator_image_small':'annotator_image'"
      :src="imdata.url"
      ref="image" 
      @load="draw_initial_contents"
    >
    <canvas
      class="annotator_canvas"
      ref="canvas"
      @click="canvas_click" 
      @mouseover="hover(true)"
      @mouseleave="hover(false)"
    />
    <div 
      class="form-check annotator-check" 
    >
      <input 
        class="form-check-input" 
        type="checkbox" 
        :disabled="read_only" 
        v-model="this.imdata.marked_accepted"
      >
      <!-- we use the save method to sync state, so we disable this for the small vue -->
    </div>
    <div
        class="form-check activation-check">
        <input
            class="form-check-input"
            type="checkbox"
            id="activation-checkbox"
            v-model="this.show_activation"
            @change="this.activation_press()"
        >
    </div>
  </div>
<!-- question: could the @load callback for img fire before created() or mounted()? (
    eg, the $refs and other vue component object attributes) -->
</template>
<script>

import paper from 'paper/dist/paper-core';

export default { 
  name: "MAnnotator", // used by ipyvue?
  props: ['initial_imdata', 'read_only'],
  emits: ['cclick'],
  data : function() {
        return {height_ratio:null, width_ratio:null, 
                paper: null, 
                imdata : this.initial_imdata,  
                show_activation : false,
                activation_paths : [], 
                activation_layer : null, }
  },
  created : function (){
      console.log('created annotator')
  },
  mounted : function() {
        this.paper = new paper.PaperScope();
        new paper.Tool(); // also implicitly adds tool to paper scope
        console.log('mounted annotator'); 
        
  },
  methods : {
    activation_press: function(){
        console.log("checks"); 
        if (this.show_activation){
            this.draw_activation(); 
        } else {
            this.clear_activation(); 
        }
    }, 
    draw_activation: function(){
        let img = this.$refs.image;         
        let container = this.$refs.container;
        
        let height = img.height;
        let width = img.width;
        // when the image has no max size, the container div 
        // ends up with a size of 0, and centering the element
        // does not seem to work
        container.style.setProperty('width', width + 'px')
        container.style.setProperty('height', height + 'px')
        img.style.setProperty('display', 'block')

        let paper = this.paper;
        paper.activate(); 

        this.activation_layer = new paper.Layer()
        this.activation_layer.activate(); 

        paper.view.draw();

        var activation = this.imdata.activation; 
        var activation = [[.5, .2, 0], 
                        [.2, .1, 0], 
                        [.1, 0, 0]]; 
        var square_width = this.$refs.image.width / activation[0].length; 
        var square_height = this.$refs.image.height / activation.length; 
        for (var x = 0; x < activation[0].length; x++){
            for (var y = 0; y < activation.length; y++){
                let paper_style = ['Rectangle', square_width * x, square_height * y, square_width, square_height];
                let rect = paper.Rectangle.deserialize(paper_style)
                let r = new paper.Path.Rectangle(rect);
                r.fillColor = 'red'; 
                r.strokeWidth = 0; 
                r.opacity = activation[y][x]
                this.activation_paths.push(r); 
            }
        } 
        
        paper.view.draw();
        paper.view.update();
      
    }, 
    clear_activation: function(){
        this.activation_layer.remove(); 
        this.activation_layer = null; 
        while (this.activation_paths.length !== 0){
            var path = this.activation_paths.pop(); 
            path.remove();
        }
    }, 
    rescale_box : function(box, height_scale, width_scale) {
          let {x1,x2,y1,y2} = box;
          return {x1:x1*width_scale, x2:x2*width_scale, y1:y1*height_scale, y2:y2*height_scale};
    },
    save : function() {
        if (this.show_activation){
            this.clear_activation(); 
        }
        let paper = this.paper
        paper.activate(); 
        let boxes = (paper.project.getItems({className:'Path'})
                          .map(x =>  {let b = x.bounds; return {x1:b.left, x2:b.right, y1:b.top, y2:b.bottom}})
                          .map(box => this.rescale_box(box, this.height_ratio, this.width_ratio)))
        console.log('saving state from paper to local imdata ')
        if (boxes.length == 0){
            this.imdata.boxes = null;
            console.log('length 0 reverts to null right now')
        } else {
            this.imdata.boxes = boxes;
        }
    },
    get_latest_imdata(){
      this.save();
      return this.imdata;
    },
    load_current_box_data : function() {
      // assumes currently image on canvas is the one where we want to display boxes on
      let paper = this.paper;
      paper.activate(); 
      // console.log('about to iterate', this);

      if (this.initial_imdata.boxes != null) {
          console.log('drawing boxes', this.initial_imdata.boxes)
        for (const boxdict of this.initial_imdata.boxes) {
            let rdict = this.rescale_box(boxdict, this.height_ratio, this.width_ratio);
            let paper_style = ['Rectangle', rdict.x1, rdict.y1, rdict.x2 - rdict.x1, rdict.y2 - rdict.y1];
            let rect = paper.Rectangle.deserialize(paper_style)
            let r = new paper.Path.Rectangle(rect);
            r.strokeColor = 'green';
            r.strokeWidth = 2;
            r.data.state = null;
            r.selected = false;
            console.log('drew rect ', r)
            }
      }
    },
    hover : function (start) {
        if (this.read_only) {
            if (start) {
                this.$refs.image.style.opacity = .5
            } else {
                this.$refs.image.style.opacity = 1.
            }
        } 
    },
    toggle_accept(){
        this.imdata.marked_accepted = this.imdata.marked_accepted ? false : true
    },
    canvas_click : function (e){
        console.log('canvas click!', e);
        this.$emit('cclick', e)
    },

    draw_initial_contents : function() {
        console.log('(draw)setting up', this)
        let img = this.$refs.image;         
        let container = this.$refs.container;
        
        let height = img.height;
        let width = img.width;
        // when the image has no max size, the container div 
        // ends up with a size of 0, and centering the element
        // does not seem to work
        container.style.setProperty('width', width + 'px')
        container.style.setProperty('height', height + 'px')
        img.style.setProperty('display', 'block')

        if (this.read_only && (this.initial_imdata.boxes === null || this.initial_imdata.boxes.length === 0)){
            return;
        }
        // call some code to draw activation array 
        // on top of canvas 
        // ctx
        // ctx = f(cnv)
        let paper = this.paper;      
        paper.activate(); 
        let cnv = this.$refs.canvas;
        console.log('drawing canvas', img.height, img.width, img)
        cnv.height = height;
        cnv.width = width;
        this.paper.setup(cnv);
        this.height_ratio = height / img.naturalHeight
        this.width_ratio = width / img.naturalWidth
        this.paper.view.draw();
        
        this.load_current_box_data();


        if (!this.read_only) {
            this.setup_box_drawing_tool(paper);
        }

        paper.view.draw();
        paper.view.update();
    },
    draw_full_frame_box(){
      // implement me
    },
    makeRect(from, to){
        this.paper.activate(); 
          let r = new this.paper.Path.Rectangle(from, to);
          r.strokeColor = 'green';
          r.strokeWidth = 2;
          r.data.state = null;
          r.selected = false;
          return r;
    },
    setup_box_drawing_tool : function(paper) {
      let tool = paper.tool;
      let makeRect = this.makeRect; // needed for => 

      tool.onMouseDown = (e) => {
          let hit_opts = {segments: true,
                          stroke: true,
                          fill: true,
                          class : paper.Path,
                          tolerance: 10};

          let hr = paper.project.hitTest(e.point, hit_opts);
          let preselected = paper.project.getSelectedItems()
          preselected.map(r => r.selected = false); // unselect previous

          let rect = null;
          if (hr == null && preselected.length > 0){
              return ;
          } else if (hr == null){
              rect = makeRect(e.point.subtract(new paper.Size(1,1)),
                              e.point);
          } else { // existing rect
            //  console.log(hr.item);
              rect = hr.item;
          }

          rect.selected=true; // select this

          if (hr != null && hr.type === 'stroke') {
              rect.data.state = 'moving';
              return;
          }

          if (hr == null || hr.type === 'segment') {
              rect.data.state = 'resizing';
              let r = rect.bounds

              const posns = [ ['getTopLeft','getTopRight'],
                        ['getBottomLeft','getBottomRight'] ];

              for (let i = 0; i < 2; i++){
                  for (let j = 0; j < 2; j++){
                      let corner_name = posns[i][j];
                      let p = r[corner_name](); // runs method to get that corner
                      if (p.isClose(e.point, hit_opts.tolerance)){
                          let opposite_corner_name = posns[(i + 1) % 2][(j + 1) % 2];
                          rect.data.from = r[opposite_corner_name]();
                          rect.data.to = r[corner_name]();
                      }
                  }
              }
          }
      }

      tool.onMouseUp = () => {
        //   this.save_current_box_data(); // auto save upon finishing changes
      };

      tool.onKeyUp = (e) => {
          let preselected = paper.project.getSelectedItems();
          if (preselected.length === 0){
              return;
          } else if (e.key === 'd') { // auto save upon deletion
              preselected.forEach(r => r.remove());
            // this.save_current_box_data();
          } else {
              return;
          }
      }

      tool.onMouseDrag = (e) => {
           // console.log('drag');
          let preselected = paper.project.getSelectedItems()
          let rect = null;
          if (preselected.length !== 1){
              return;
          } else{
              rect = preselected[0];
          }

          if (rect.data.state === 'moving') {
              rect.position = rect.position.add(e.point).subtract(e.lastPoint);
          } else if (rect.data.state === 'resizing') {
              let bounds = new paper.Rectangle(rect.data.from, e.point);
              if (bounds.width !== 0 && bounds.height !== 0){
                  rect.bounds = bounds;
              }
          }
      };
    }
    }
}
</script>
<style scoped>
.activation-check {
    position: absolute; 
    bottom: 10px; 
    left: 10px; 
}

.annotator_div {
    position:relative;
    margin:0px;
    border:0px;
    padding:0px;
    /* width:fit-content; set dynamically after image is loaded */
    /* height:fit-content; */
    /* display:inline-block;*/  /* let this decision be done elsewhere, just like with an image */
}


.annotator_image {
    /* max-width:100%; */
    /* max-height:100%; */
    display: none;
    position:absolute;
    top:0px;
    left:0px;
    margin:0px;
    border:0px;
    padding:0px;
    object-fit:none; /* never rescale up */ 
}

.annotator-check {
  position:absolute;
  top: 10px;
  left: 10px;
}

.annotator_image_small {
  width: auto !important;
  height: auto !important;
  margin:0px;
  border:0px;
  padding:0px;
  max-height: 200px;
  /*max-width: 400px; */
  object-fit: scale-down; /* never rescale up */ 
}

.annotator_canvas {
    /* border:0px solid #d3d3d3; */
    /* max-width:100%;*/
    /* max-height:100%; */
    position:absolute; 
    top:0px; 
    left:0px;
    margin:0px;
    border:0px;
    padding:0px;
    /* display:;  */
    /* for now not showing it to try to fix centering issue...*/
}

</style>