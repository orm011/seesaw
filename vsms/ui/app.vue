<template>
<div>
  <!-- adapted from https://github.com/twbs/bootstrap/blob/main/site/content/docs/5.0/examples/dashboard/index.html  -->
  <header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
        <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">VSL</a>
        <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" 
        data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" 
        aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <input class="form-control form-control-dark w-100" type="text" placeholder="Search" aria-label="Search" 
          v-model="text_query" v-on:keydown.enter="text(text_query)"/>
        <div class="navbar-nav col-lg-1 px-3"/>
  </header>
  <div class="container-fluid">
    <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
      <div class="position-sticky pt-3">
        <div class='row'>
          <button class="btn btn-dark btn-block" @click="reset()"> Reset </button>
        </div>
        <div class='row'>
        <ul class="nav flex-column">
          <li v-for="(dataset_name,idx) in data.datasets" :key="idx" class="nav-item">
            <a  :class="`nav-link ${(dataset_name === data.current_dataset) ? 'active' : ''}`" aria-current="page" href="#" 
              @click="reset(dataset_name)">
              {{dataset_name}}
            </a>
          </li>
        </ul>
        </div>
      </div>
    </nav>
    <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
        <div class="row">
          <m-image-gallery v-if="data.image_urls.length > 0" :image_urls="data.image_urls" :ldata="data.ldata" 
            v-on:update:selection="onselection($event)" />
        </div>
        <div class="row">
              <button v-if="data.image_urls.length > 0" @click="next()" class="btn btn-dark btn-block">More...</button>
          </div>
          <!-- <m-annotator v-if="selection!=null" :image_url="data.image_urls[selection]" :adata="data.ldata[selection]" /> -->
    </main>
    </div> 
 </div>
</template>
<script>
import MImageGallery from './m-image-gallery.vue';
import MAnnotator from './m-annotator.vue';

export default {
    components : {'m-annotator':MAnnotator, 'm-image-gallery':MImageGallery},
    // ldata : [{'value': -1, 'id': int, 'dbidx': int, 'boxes':[]}]
    data () { return {  data:{image_urls:[], ldata:[]}, selection: null, datasets:[], current_dataset:'coco'} },
    mounted (){
        fetch('/vlsapi/getstate', {cache: "reload"})
            .then(response => response.json())
            .then(data => (this.data = data, this.selection=null))
    },
    methods : {
        reset(dsname){
          console.log(dsname)
          let reqdata = {todataset:dsname};
          console.log(reqdata);

            fetch(`/vlsapi/reset`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(reqdata)
            })
            .then(response => response.json())
            .then(data => (this.data = data, this.selection = null))
        },
        text(text_query){
            fetch(`/vlsapi/text?key=${encodeURIComponent(text_query)}`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(this.data) // body data type must match "Content-Type" header
            })
            .then(response => response.json())
            .then(data => (this.data = data, this.selection = null))
        },
        next(){
            fetch(`/vlsapi/next`, {method:'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(this.data) // body data type must match "Content-Type" header
                            })
            .then(response => response.json())
            .then(data => (this.data = data, this.selection = null))
        },
        onselection(ev){
          const next_value = [1,1,0][this.data.ldata[ev].value+1]; // {-1:1, 0:1,1:0}
          this.$set(this.data.ldata[ev], 'value', next_value);

          if (next_value === 1){
            this.$set(this.data.ldata[ev], 'boxes', [{'xmin':0, 'xmax':1, 'ymin':0, 'ymax':1}])
          } else if (next_value === 0) {
            this.$set(this.data.ldata[ev], 'boxes', [])
          } else if (next_value === -1) {
            this.$set(this.data.ldata[ev], 'boxes', null)
          }
        }
    }
}
</script>
<style scoped>
/* https://raw.githubusercontent.com/twbs/bootstrap/main/site/content/docs/5.0/examples/dashboard/dashboard.css */
body {
  font-size: .875rem;
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
</style>