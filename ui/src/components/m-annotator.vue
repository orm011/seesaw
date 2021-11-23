<template>
    <div class="annotator_div" ref="container" @keyup.esc="emit('esc')" tabindex='0'>
        <img :class="read_only ? 'annotator_image_small':'annotator_image'" :src="initial_imdata.url" ref="image" 
                @load="draw_initial_contents" tabindex='1' @keyup.esc="emit('esc')" />
        <canvas class="annotator_canvas" ref="canvas" @keyup.esc="emit('esc')" tabindex='2' @click="canvas_click" 
                @mouseover="hover(true)" @mouseleave="hover(false)" />
    </div>
<!-- question: could the @load callback for img fire before created() or mounted()? (
    eg, the $refs and other vue component object attributes) -->
</template>
<script>

import paper from 'paper/dist/paper-core';

export default { 
  name: "m-annotator", // used by ipyvue?
  props: ['initial_imdata', 'read_only'],
  data : function() {
        return {height_ratio:null, width_ratio:null, paper: null }
  },
  created : function (){
      console.log('created annotator')
  },
  mounted : function() {
        this.paper = new paper.PaperScope();
        new paper.Tool(); // also implicitly adds tool to paper scope
        console.log('mounted annotator')
  },
  methods : {
    rescale_box : function(box, height_scale, width_scale) {
          let {x1,x2,y1,y2} = box;
          return {x1:x1*width_scale, x2:x2*width_scale, y1:y1*height_scale, y2:y2*height_scale};
    },
    save_current_box_data : function() {
        let paper = this.paper
        let boxes = (paper.project.getItems({className:'Path'})
                          .map(x =>  {let b = x.bounds; return {x1:b.left, x2:b.right, y1:b.top, y2:b.bottom}})
                          .map(box => this.rescale_box(box, this.height_ratio, this.width_ratio)))
        console.log('saving boxes', )
        if (boxes.length == 0){
            console.log('length 0 reverts to null right now')
            this.$emit('box-save', null)
        } else {
            this.$emit('box-save', boxes)
        }
    },
    load_current_box_data : function() {
      // assumes currently image on canvas is the one where we want to display boxes on
      let paper = this.paper;
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
    canvas_click : function (e){
        console.log('canvas click!', e);
        this.$emit('cclick', e)
    },
    draw_initial_contents : function() {
        console.log('(draw)setting up', this)
        let paper = this.paper;
        let img = this.$refs.image; 
        let cnv = this.$refs.canvas;

        // size of element outside
        cnv.height = img.height;
        cnv.width = img.width;

        paper.setup(cnv);
        this.height_ratio = img.height / img.naturalHeight
        this.width_ratio = img.width / img.naturalWidth
        paper.view.draw();
        this.load_current_box_data();

        if (!this.read_only) {
            this.setup_box_drawing_tool(paper);
        }

        paper.view.draw();
        paper.view.update();
    },
    setup_box_drawing_tool : function(paper) {
      let tool = paper.tool;
      let makeRect = (from,to) => {
          let r = new paper.Path.Rectangle(from,to);
          r.strokeColor = 'green';
          r.strokeWidth = 2;
          r.data.state = null;
          r.selected = false;
          return r;
      }

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
          this.save_current_box_data(); // auto save upon finishing changes
      };

      tool.onKeyUp = (e) => {
          let preselected = paper.project.getSelectedItems();
          if (preselected.length === 0){
              return;
          } else if (e.key === 'd') { // auto save upon deletion
              preselected.forEach(r => r.remove());
              this.save_current_box_data();
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
.annotator_div {
    position:relative;
    margin:0px;
    /* width:fit-content;
    height:fit-content; */
    display:inline-block
}
.annotator_image {
    /* max-width:100%; */
    /* max-height:100%; */
    /* display:inline; */
    position:relative;
    object-fit:none; /* never rescale up */ 
}
.annotator_image_small {
  width: auto !important;
  height: auto !important;
  max-height: 200px;
  /*max-width: 400px; */
  object-fit: scale-down; /* never rescale up */ 
}

.annotator_canvas {
    /* border:0px solid #d3d3d3; */
    /* max-width:100%;*/
    /* max-height:100%; */
    position:absolute; top:0px; left:0px;
    /* display:;  */
    /* for now not showing it to try to fix centering issue...*/
}

</style>