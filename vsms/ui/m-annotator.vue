<template>
    <div class="annotator_div" ref="container">
        <canvas resize class="annotator_canvas" ref="canvas" height="$refs.image.naturalHeight"
        width="$refs.image.naturalWidth"/>
        <img class="annotator_image" :src="image_url" ref="image" @load="draw_initial_contents" />
    </div>
<!-- question: could the @load callback for img fire before created() or mounted()? (
    eg, the $refs and other vue component object attributes) -->
</template>
<style scoped>
.annotator_div {
    margin:1px;
    width:600px;
    height:600px;
}
.annotator_image {
    /* max-width:100%; */
    /* max-height:100%; */
    display:none;
    object-fit:scale-down; /* never rescale up */ 
}
.annotator_canvas {
    /* border:0px solid #d3d3d3; */
    /* max-width:100%;*/
    /* max-height:100%; */
    display:inline-block;
}

</style>
<script>
export default { 
  name: "m-annotator.vue", // used by ipyvue?
  props: ['image_url', 'adata', 'read_only'],
  data : function() {
        let paper = new window.paper.PaperScope();
        new paper.Tool(); // also implicitly adds tool to paper scope
        return {height_ratio:null, width_ratio:null, paper:paper}
  },
  created : function (){
  },
  methods : {
    rescale_box : function(box, height_scale, width_scale) {
          let {xmin,xmax,ymin,ymax} = box;
          return {xmin:xmin*width_scale, xmax:xmax*width_scale, ymin:ymin*height_scale, ymax:ymax*height_scale};
    },
    save_current_box_data : function() {
        let paper = this.paper
        let boxes = (paper.project.getItems({className:'Path'})
                          .map(x =>  {let b = x.bounds; return {xmin:b.left, xmax:b.right, ymin:b.top, ymax:b.bottom}})
                          .map(box => this.rescale_box(box, this.height_ratio, this.width_ratio)))
        this.adata.boxes = boxes;
    },
    load_current_box_data : function() {
      // assumes currently image on canvas is the one where we want to display boxes on
      let paper = this.paper;
      // console.log('about to iterate', this);
      for (const boxdict of this.adata.boxes) {
          let rdict = this.rescale_box(boxdict, 1./this.height_ratio, 1./this.width_ratio);
          let paper_style = ['Rectangle', rdict.xmin, rdict.ymin, rdict.xmax - rdict.xmin, rdict.ymax - rdict.ymin];
          let rect = paper.Rectangle.deserialize(paper_style)
          let r = new paper.Path.Rectangle(rect);
          r.strokeColor = 'red';
          r.strokeWidth = 1;
          r.data.state = null;
          r.selected = false;
        }
    },
    draw_initial_contents : function() {
        console.log('setting up', this)
        let paper = this.paper;
        let img = this.$refs.image; 
        let cnv = this.$refs.canvas; 

        let scale = Math.min(img.width/img.naturalWidth, img.height/img.naturalHeight);
        let img_height = Math.round(scale*img.height);
        let img_width = Math.round(scale*img.width);

        // size of element outside
        cnv.style.height = img_height + 'px';
        cnv.style.width = img_width + 'px';

        paper.setup(cnv);
        let raster = new paper.Raster(img);

        // Move raster to view center
        raster.position = paper.view.center;
        raster.size = paper.view.bounds;
        // raster.fitBounds(paper.view.bounds)
        // paper.view.onResize = (e) => { // used for responsive resizing. would need to also resize the boxes...
        //     // console.log('resized! new aspect ratio:', e.width, e.height, e.width/e.height);
        // }
        raster.locked = true;
        this.height_ratio = img.height / raster.height;
        this.width_ratio = img.width / raster.width;
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
          r.strokeColor = 'red';
          r.strokeWidth = 1;
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
          } else if (e.key === 't') { // auto save upon deletion
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