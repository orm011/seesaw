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
        disabled 
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
  emits: ['cclick', 'selection'],
  data : function() {
        return {height_ratio:null, width_ratio:null, 
                paper: null, 
                imdata : this.initial_imdata,
                show_activation : false,
                annotation_paper_objs : [], // {box, description}
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
    toggle_activation() {
        this.show_activation = !this.show_activation
        this.activation_press();
    },
    delete_paper_obj(obj){
        this.annotation_paper_objs = this.annotation_paper_objs.filter((oent) => (oent != obj))
        obj.box.remove()
        obj.description.remove()
        this.save()
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

        let layer = paper.project.activeLayer; 

        this.activation_layer = new paper.Layer({locked: true}); 
        paper.project.insertLayer(0, this.activation_layer); 
        this.activation_layer.activate(); 

        paper.view.draw();

        console.log("activations"); 
        console.log(this.initial_imdata.activations); 
        for (let b of this.initial_imdata.activations){
          let boxdict = b.box
          let rdict = this.rescale_box(boxdict, this.height_ratio, this.width_ratio);
          let paper_style = ['Rectangle', rdict.x1, rdict.y1, rdict.x2 - rdict.x1, rdict.y2 - rdict.y1];
          let rect = paper.Rectangle.deserialize(paper_style)
          let r = new paper.Path.Rectangle(rect);
          if (b.score >= 0){
            r.fillColor = 'red'; 
            r.strokeWidth = 0; 
            r.opacity = b.score
          } else { // helpful to visualize negative 
            r.fillColor = 'blue'
            r.strokeWidth = 0; 
            r.opacity = -b.score
          }
          this.activation_paths.push(r); 
        }

        layer.activate(); 
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
          let {x1,x2,y1,y2,...rest} = box;
          return {x1:x1*width_scale, x2:x2*width_scale, y1:y1*height_scale, y2:y2*height_scale, ...rest};
    },

    /**
     * @param {{box:paper.Path, description : paper.PointText}} obj
     */
    paper2imdata(obj){
      let b = obj.box.bounds;
      let ret = {x1:b.left, x2:b.right, y1:b.top, y2:b.bottom}
      let ans = this.rescale_box(ret, this.height_ratio, this.width_ratio)

      ans.description = obj.description.content;
      ans.marked_accepted = 'marked_accepted' in obj.box.data ?  obj.box.data.marked_accepted  : false
      return ans
    },
    save : function() {
        let paper = this.paper
        paper.activate(); 

        let boxes = this.annotation_paper_objs.map(this.paper2imdata);
        console.assert(boxes != undefined)
        console.log('saving state from paper to local imdata ')
        if (boxes.length == 0) {
            this.imdata.boxes = null;
            this.imdata.marked_accepted = false;
            console.log('length 0 reverts to null right now')
        } else {
            this.imdata.boxes = boxes;
            this.imdata.marked_accepted = boxes.filter(b => b.marked_accepted).length > 0
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
            r.data.marked_accepted = boxdict.marked_accepted;
            r.strokeColor = boxdict.marked_accepted ? 'green' : 'yellow';
            r.strokeWidth = 4;
            r.data.state = null;
            r.selected = false;

            let text = new paper.PointText(new paper.Point(rdict.x1, rdict.y1));
            text.justification = 'left';
            text.fillColor = r.strokeColor;
            text.fontSize = 12; // default 10
            text.content = boxdict.description;

            let annot_obj = {box:r, description:text};
            this.annotation_paper_objs.push(annot_obj);
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
          r.strokeWidth = 4;
          r.data.state = null;
          r.data.marked_accepted = false;
          r.strokeColor = r.data.marked_accepted ? 'green' : 'yellow'
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
            console.log('mouse down')
          let hr = paper.project.hitTest(e.point, hit_opts);
          let preselected = paper.project.getSelectedItems()
          preselected.map(r => r.selected = false); // unselect previous

          let rect = null;
          if (hr == null && preselected.length > 0){
              return ;
          } else if (hr == null){ // make a new one
              rect = makeRect(e.point.subtract(new paper.Size(1,1)),
                              e.point);
              let text = new paper.PointText(e.point);
              text.justification = 'left';
              text.fillColor = rect.data.marked_accepted ? 'green' : 'yellow'
              text.content = ''
              let sel = {box:rect, description:text};
              this.annotation_paper_objs.push(sel)
          } else { // existing rect
              rect = hr.item;
          }

          rect.selected=true; // mark selected
          let cur_sel = this.annotation_paper_objs.filter((obj) => obj.box.selected)
          if (cur_sel.length == 1){
            // this.$emit('selection', cur_sel[0]);
          } else {
            console.log('multiple/zero selected items')
          }

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
          console.log("Mouse up"); 
          this.save() //_current_box_data(); // auto save upon finishing changes
          
          let sels = this.annotation_paper_objs.filter(obj => obj.box.selected)
          if (sels.length === 1){
            this.$emit('selection', sels[0])
          } else {
            this.$emit('selection', null)
          }
      };

      tool.onKeyUp = (e) => {
        //   if (this.being_edited == null){
        //   let preselected = paper.project.getSelectedItems();
        //   if (preselected.length === 0){
        //       return;
        //   } else if (e.key === 'd') { // auto save upon deletion
        //       preselected.forEach(r => r.remove());
        //       this.save(); // want to know if mark accept or not
        //     // this.save_current_box_data();
        //   } 
        //   else if (e.key == 'enter'){
        //       // if (preselected.length == 1){
        //       //   let item = preselected[0];
        //       //   let matched = this.annotation_paper_objs.filter(x => x.box == item);
        //       //   if (matched.length == 1){
        //       //       this.being_edited = matched[0];
        //       //   }
        //       // } else{
        //       //   console.log('need at least one selected')
        //       // }
        //   }
        //    else {
        //       return;
        //   }
        // } else{ 
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

    console.log('finished setting up box annotation tool ')
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