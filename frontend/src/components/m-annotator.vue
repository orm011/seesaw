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
<script lang="ts">

import {defineComponent} from 'vue';
import paper from 'paper/dist/paper-core';
import {image_accepted} from '../util';
import {Imdata, Box} from '../basic_types';

interface PaperObject {
  box : paper.Path.Rectangle,
  description : paper.TextItem
}

export default defineComponent({ 
  name: "MAnnotator", // used by ipyvue?
  props: ['initial_imdata', 'read_only', 'front_end_type'],
  emits: ['cclick', 'selection'],
  data : function() {
        return {height_ratio:null, width_ratio:null, 
                paper: new paper.PaperScope(), 
                imdata : this.initial_imdata as Imdata,
                show_activation : false,
                annotation_paper_objs : [] as PaperObject[],
                activation_paths : [], 
                activation_layer : null,
                text_offset : null, 
                text_dict: {}, 
                show_float: true}
  },
  created : function (){
      console.log('created annotator')
  },
  mounted : function() {
        new paper.Tool(); // also implicitly adds tool to paper scope
        this.text_offset = new this.paper.Point(5, 20); 
        console.log('mounted annotator'); 
        
  },
  methods : {
    image_accepted(imdata : Imdata): boolean{ // make it accessible from the <template>
          return image_accepted(imdata)
    },
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
    delete_paper_obj(obj : PaperObject){
        this.annotation_paper_objs = this.annotation_paper_objs.filter((oent) => (oent != obj))
        obj.box.remove()
        obj.description.remove()
        this.save()
    },
    draw_activation: function(){
        console.log("Beginning"); 
        let img = this.$refs.image as HTMLImageElement;         
        let container = this.$refs.container as HTMLDivElement;
        
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

        let layer = null; 
        console.log("before if"); 
        if (paper.project !== null){
            console.log("not null"); 
            layer = paper.project.activeLayer; 
        }
        console.log("after if"); 

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
            r.fillColor = 'yellow'; 
            r.strokeWidth = 0; 
            r.opacity = b.score
          } else { // helpful to visualize negative 
            r.fillColor = 'blue'; 
            r.strokeWidth = 0; 
            r.opacity = -b.score
          }
          this.activation_paths.push(r); 
          if (this.show_float){
            let strokeColor = 'black'
            let point = new paper.Point(rdict.x1 + this.text_offset.x, rdict.y1 + this.text_offset.y); 
            let text = new paper.PointText(point);
            text.justification = 'left';
            text.fillColor = strokeColor;
            text.fontSize = 12; // default 10
            text.content = b.score.toFixed(2);

            this.activation_paths.push(text); 
          }
        }

        if (layer !== null){
            layer.activate(); 
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
    rescale_box : function(box : Box, height_scale : number, width_scale : number) {
          let {x1,x2,y1,y2,...rest} = box;
          return {x1:x1*width_scale, x2:x2*width_scale, y1:y1*height_scale, y2:y2*height_scale, ...rest};
    },

    /**
     * @param {{box:paper.Path, description : paper.PointText}} obj
     */
    paper2imdata(obj : PaperObject){
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
            this.draw_box(boxdict, paper); 
          }
      }
    },

    draw_box : function(boxdict : Box, paper : paper.PaperScope) {
        let strokeColor = boxdict.marked_accepted ? 'green' : 'red';
        let rdict = this.rescale_box(boxdict, this.height_ratio, this.width_ratio);
        let paper_style = ['Rectangle', rdict.x1, rdict.y1, rdict.x2 - rdict.x1, rdict.y2 - rdict.y1];
        let rect = paper.Rectangle.deserialize(paper_style); 
        let r = new paper.Path.Rectangle(rect);
        r.data.marked_accepted = boxdict.marked_accepted;
        r.strokeColor = strokeColor; 
        r.strokeWidth = 4;
        r.data.state = null;
        r.selected = false;

        let point = new paper.Point(rdict.x1 + this.text_offset.x, rdict.y1 + this.text_offset.y); 
        let text = new paper.PointText(point);
        text.justification = 'left';
        text.fillColor = strokeColor;
        text.fontSize = 12; // default 10
        text.content = boxdict.description;

        this.text_dict[r] = text; 
        let annot_obj = {box:r, description:text};
        this.annotation_paper_objs.push(annot_obj);
        return annot_obj;
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
        let img = this.$refs.image as HTMLImageElement;         
        let container = this.$refs.container as HTMLDivElement;
        
        let height = img.height;
        let width = img.width;
        // when the image has no max size, the container div 
        // ends up with a size of 0, and centering the element
        // does not seem to work
        container.style.setProperty('width', width + 'px')
        container.style.setProperty('height', height + 'px')
        img.style.setProperty('display', 'block')

        // call some code to draw activation array 
        // on top of canvas 
        // ctx
        // ctx = f(cnv)
        let paper = this.paper;      
        paper.activate(); 
        let cnv = this.$refs.canvas as HTMLCanvasElement;
        console.log('drawing canvas', img.height, img.width, img)
        cnv.height = height;
        cnv.width = width;
        this.paper.setup(cnv);
        this.height_ratio = height / img.naturalHeight
        this.width_ratio = width / img.naturalWidth
        this.paper.view.draw();
        
        if (this.read_only && (this.initial_imdata.boxes === null || this.initial_imdata.boxes.length === 0)){
            return;
        }
        this.load_current_box_data();


        if (!this.read_only && (this.front_end_type !== 'plain')) {
            this.setup_box_drawing_tool(paper);
        }

        paper.view.draw();
        paper.view.update();
    },
    draw_full_frame_box(accepted){
      // implement me
      let paper = this.paper; 
      paper.activate(); 

      let preselected = paper.project.getSelectedItems()
      preselected.map(r => r.selected = false); // unselect previous

      let draw = true; 
      let boxes = this.annotation_paper_objs.map(this.paper2imdata);
      let img = this.$refs.image;         
      let height = img.height;
      let width = img.width;

      for (var index in boxes){
        let box = boxes.at(index); 
        console.log(box); 
        if (box.x1 == 0 && box.x2 == width && box.y1 == 0 && box.y2 == height){
          draw = false; 
          console.log("Box not drawn, box returned"); 
          console.log(box); 
          let object = this.annotation_paper_objs.at(index);
          console.log(object);  
          object.box.selected = true; 
        }
      }
      
      if (draw){
        let boxdict = {x1: 0, x2: width, y1: 0, y2: height, description: '', marked_accepted: accepted}
        let box = this.draw_box(boxdict, paper); 
        console.log(box); 
        box.box.selected=true;
      }

      this.save(); 
      let sels = this.annotation_paper_objs.filter(obj => obj.box.selected)
      this.full_box = sels[0]; 
      if (sels.length === 1){
        this.$emit('selection', sels[0])
      }

      /**
       * TODO: 
       * Deselect all selected items already
       * indices of boxes and annotation objects are the same, use boxes and then return annotation objects indexed
       */

    },
    makeRect(from, to){
        this.paper.activate(); 
          let r = new this.paper.Path.Rectangle(from, to);
          r.strokeWidth = 4;
          r.data.state = null;
          if (this.front_end_type === 'plain' || this.front_end_type === 'pytorch'){
            r.data.marked_accepted = true; 
          } else {
            r.data.marked_accepted = false;
          }
          r.strokeColor = r.data.marked_accepted ? 'green' : 'red'
          r.selected = false;
          return r;
    },
    redraw_text : function(box) {
      let old_text = this.text_dict[box]; 
      let paper = this.paper; 
      this.paper.activate(); 

      let point = new paper.Point(box.bounds.x + this.text_offset.x, box.bounds.y + this.text_offset.y); 
      old_text.point = point; 
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
              console.log("Point: ", e.point);
              let point = new paper.Point(e.point.x + this.text_offset.x, e.point.y + this.text_offset.y); 
              console.log("New Points:", point);  
              let text = new paper.PointText(point);
              text.justification = 'left';
              text.fillColor = rect.data.marked_accepted ? 'green' : 'red'
              text.content = ''
              this.text_dict[rect] = text; 
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
          this.redraw_text(rect);
      };

    console.log('finished setting up box annotation tool ')
    }
    }
})
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