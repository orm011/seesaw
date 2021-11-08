<template>
<div>
  <!-- adapted from https://github.com/twbs/bootstrap/blob/main/site/content/docs/5.0/examples/dashboard/index.html  -->
  <header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
        <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">SeeSaw</a>
        <input class="form-control form-control-dark w-auto" type="text" placeholder="Search" aria-label="Search" 
          v-model="text_query" v-on:keydown.enter="text(text_query)"/>
        <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" 
        data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" 
        aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="navbar-nav col-lg-1 px-3"/>
  </header>
  <div class="container-fluid">
    <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
      <div class="position-sticky pt-3">
        <div class='row'>
          <div class="col">
          <div class='row'>
          <label>Current Database:</label>
          </div>
          <div class='row'>
          <select v-model="current_dataset" v-on:change="reset(current_dataset)">
            <option v-for="(dsname,idx) in datasets" :key="idx" :value="dsname">{{dsname}}</option>
          </select>
          </div>
          </div>
        </div>
        <div class='row'>
          <span>Total images seen: {{total_images()}}</span>
        </div>
        <div class='row'>
          <span>Total results found: {{total_annotations()}}</span>
        </div>
        <div class='row'>
          <button class="btn btn-dark btn-block" @click="save()"> Save </button>
        </div>
        <!-- <div class='row'>
        <ul class="nav flex-column">
          <li v-for="(dataset_name,idx) in datasets" :key="idx" class="nav-item">
            <a  :class="`nav-link ${(dataset_name === current_dataset) ? 'active' : ''}`" aria-current="page" href="#" 
              @click="reset(dataset_name)">
              {{dataset_name}}
            </a>
          </li>
        </ul>
        </div> -->
        <div v-if="refmode" class='row'>
          <label for="reference category">(DEBUG) pick ground truth category:</label>
          <select v-model="current_category">
            <option v-for="(cat,idx) in ['', ...reference_categories]" :key="idx" :value="cat">{{cat}}</option>
          </select>
        </div>
      </div>
    </nav>
    <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
        <div class="row" v-for="(imdata,idx) in gdata" :key="idx">
          <div v-if="timing.length > 0" class="row">
            <span>Search refinement took {{timing[idx].toFixed(2)}} seconds</span>
          </div>
          <div class="row">
          <m-image-gallery ref="galleries" v-if="imdata.length > 0" :initial_imdata="filter_boxes(imdata, current_category)"
              v-on:data_update="data_update(idx, $event)" :refmode="refmode" v-on:copy-ref="copy_ref(idx, $event)"/>
          </div>
          <div class="row space"/>
        </div>
        <div class="row" v-if="gdata.length > 0">
              <button @click="next()" class="btn btn-dark btn-block">More...</button>
        </div>
    </main>
    </div> 
 </div>
</template>
<script>
import MImageGallery from './m-image-gallery.vue';
import MAnnotator from './m-annotator.vue';

export default {
    components : {'m-annotator':MAnnotator, 'm-image-gallery':MImageGallery},
    props: {},
    data () { return {  
                gdata:[],  // list of lists of {'url': str, 'dbidx': int, 'boxes':null|List, 'refboxes':null|List}
                timing:[],
                selection: null, 
                datasets:[''], 
                current_dataset:'', 
                reference_categories:[],
                current_category:'', 
                text_query:null,
                refmode : false,
              }
            },
    mounted (){
        fetch('/api/getstate', {cache: "reload"})
            .then(response => response.json())
            .then(data => (this.gdata = data.gdata,
                  this.datasets = data.datasets, 
                  this.reference_categories = data.reference_categories,
                  this.current_dataset = data.current_dataset )
            )
    },
    methods : {
        total_images() {
            return this.gdata.map((l) => l.length).reduce((a,b)=>a+b, 0)
        },
        total_annotations(){
            let annot_per_list = function(l){
              let total = 0;
              for (const elt of l){
                if (elt.boxes == null){
                  continue;
                } else {
                  total += elt.boxes.length;
                }
              }
              return total;
            };
            return this.gdata.map(annot_per_list).reduce((a,b)=>a+b, 0)
        },
        filter_boxes(imdata, category){
          let out = []
          for (const ent of imdata){
            let outent = {...ent};
            outent.refboxes = ent.refboxes.filter((b) => b.category === category || category === '');
            out.push(outent);
          }
          return out
        },
        data_update(gdata_idx, ev){
          console.log('data_update')
          this.gdata[gdata_idx][ev.idx].boxes =  ev.boxes
        },
        copy_ref(gdata_idx, panel_idx){
          let refboxes = this.gdata[gdata_idx][panel_idx].refboxes;
          let frefboxes = refboxes.filter((b) => b.category === this.current_category || this.current_category === '')
          let newboxes = frefboxes.length == 0 ? null : frefboxes
          this.gdata[gdata_idx][panel_idx].boxes = newboxes
        },
        reset(dsname){
          console.log(this);
          console.log(dsname)
          let reqdata = {dataset:dsname};
          console.log(reqdata);

            fetch(`/api/reset`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(reqdata)
            })
            .then(response => response.json())
            .then(data => (this.gdata = data.gdata, 
            this.datasets=data.datasets, 
            this.refine_text = '', 
            this.current_dataset=data.current_dataset, 
            this.reference_categories = data.reference_categories,
            this.current_category = '',
            this.text_query = null,
            this.selection = null))
        },
        text(text_query){
            fetch(`/api/text?key=${encodeURIComponent(text_query)}`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify({})}
                )
            .then(response => response.json())
            .then(data => (this.timing.push(data.time), this.gdata = data.client_data.gdata, this.selection = null))
        },
        next(){
          console.log(' this' , this);
          let body = { client_data : this.$data };

            fetch(`/api/next`, {method:'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(body) // body data type must match "Content-Type" header
                            })
            .then(response => response.json())
            .then(data => (this.timing.push(data.time), this.gdata = data.client_data.gdata, this.selection = null))
        },
        save(){
          let body = { client_data : this.$data};
          fetch(`/api/save`, {method:'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(body) // body data type must match "Content-Type" header
                            })
            .then(response => response.json())
            .then(data => (this.gdata = data.gdata, this.selection = null))
        }
    }
}
</script>
<style scoped>
/* https://raw.githubusercontent.com/twbs/bootstrap/main/site/content/docs/5.0/examples/dashboard/dashboard.css */
body {
  font-size: .875rem;
}

img {
  object-fit: scale-down;
}

.drag-el img {
  max-height: 220px;
}

.feather {
  width: 16px;
  height: 16px;
  vertical-align: text-bottom;
}

/*
 * Sidebar
 */

.sidebar {
  position: fixed;
  top: 0;
  /* rtl:raw:
  right: 0;
  */
  bottom: 0;
  /* rtl:remove */
  left: 0;
  z-index: 100; /* Behind the navbar */
  padding: 48px 0 0; /* Height of navbar */
  box-shadow: inset -1px 0 0 rgba(0, 0, 0, .1);
}

@media (max-width: 767.98px) {
  .sidebar {
    top: 5rem;
  }
}

.row .space {
    height: 5px;
    background-color: gray;
}

.sidebar-sticky {
  position: relative;
  top: 0;
  height: calc(100vh - 48px);
  padding-top: .5rem;
  overflow-x: hidden;
  overflow-y: auto; /* Scrollable contents if viewport is shorter than content. */
}

.sidebar .nav-link {
  font-weight: 500;
  color: #333;
}

.sidebar .nav-link .feather {
  margin-right: 4px;
  color: #727272;
}

.sidebar .nav-link.active {
  color: #007bff;
}

.sidebar .nav-link:hover .feather,
.sidebar .nav-link.active .feather {
  color: inherit;
}

.sidebar-heading {
  font-size: .75rem;
  text-transform: uppercase;
}

/*
 * Navbar
 */

.navbar-brand {
  padding-top: .75rem;
  padding-bottom: .75rem;
  font-size: 1rem;
  background-color: rgba(0, 0, 0, .25);
  box-shadow: inset -1px 0 0 rgba(0, 0, 0, .25);
}

.navbar .navbar-toggler {
  top: .25rem;
  right: 1rem;
}

.navbar .form-control {
  padding: .75rem 1rem;
  border-width: 0;
  border-radius: 0;
}

.form-control-dark {
  color: #fff;
  background-color: rgba(255, 255, 255, .1);
  border-color: rgba(255, 255, 255, .1);
}

.form-control-dark:focus {
  border-color: transparent;
  box-shadow: 0 0 0 3px rgba(255, 255, 255, .25);
}

  .drop-zone {
    background-color: #eee;
    margin-bottom: 10px;
    padding: 10px;
  }

  .drag-el {
    background-color: #fff;
    margin-bottom: 10px;
    padding: 5px;
  }
  
</style>