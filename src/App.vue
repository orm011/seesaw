<template>
  <div>
    <!-- adapted from https://github.com/twbs/bootstrap/blob/main/site/content/docs/5.0/examples/dashboard/index.html  -->
    <header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
      <a
        class="navbar-brand col-md-3 col-lg-2 me-0 px-3"
        href="#"
      >SeeSaw</a>
      <input
        class="form-control form-control-dark w-auto"
        type="text"
        placeholder="Search"
        aria-label="Search" 
        v-model="text_query"
        @keydown.enter="text(text_query)"
      >
      <button
        class="navbar-toggler position-absolute d-md-none collapsed"
        type="button" 
        data-bs-toggle="collapse"
        data-bs-target="#sidebarMenu"
        aria-controls="sidebarMenu" 
        aria-expanded="false"
        aria-label="Toggle navigation"
      >
        <span class="navbar-toggler-icon" />
      </button>
      <div class="navbar-nav col-lg-1 px-3" />
    </header>
    <div class="container-fluid">
      <nav
        id="sidebarMenu"
        class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse"
      >
        <div class="position-sticky pt-3">
          <div class="row">
            <div class="col">
              <div class="row">
                <label>Current Database:</label>
              </div>
              <div class="row">
                <select
                  v-model="current_index"
                  @change="reset(current_index)"
                >
                  <option
                    v-for="(idxspec,idx) in client_data.indices"
                    :key="idx"
                    :value="idxspec"
                  >
                    {{ idxspec }}
                  </option>
                </select>
              </div>
            </div>
          </div>
          <div class="row">
            <span>Total images shown: {{ total_images() }}</span>
          </div>
          <div class="row">
            <span>Total images accepted: {{ total_accepted() }}</span>
          </div>
          <div class="row">
            <button 
              class="btn btn-dark btn-block" 
              @click="save()"
            > 
              Save 
            </button>
          </div>
          <div class="row">
            <button
              class="btn btn-dark btn-block"
              @click="reset(current_index)"
            >
              Reset
            </button>
          </div>

          <div class="row">
            <input
              v-model="session_path" 
              placeholder="session path"
            >
          </div>
          <div class="row">
            <button 
              class="btn btn-dark btn-block" 
              @click="load_session(session_path)"
            > 
              Load Session
            </button>
          </div>

          <!-- <div class='row'>
        <ul class="nav flex-column">
          <li v-for="(dataset_name,idx) in indices" :key="idx" class="nav-item">
            <a  :class="`nav-link ${(dataset_name === current_index) ? 'active' : ''}`" aria-current="page" href="#" 
              @click="reset(dataset_name)">
              {{dataset_name}}
            </a>
          </li>
        </ul>
        </div> -->
          <div
            v-if="refmode"
            class="row"
          >
            <label for="reference category">(DEBUG) pick ground truth category:</label>
            <select v-model="current_category">
              <option
                v-for="(cat,idx) in ['', ...reference_categories]"
                :key="idx"
                :value="cat"
              >
                {{ cat }}
              </option>
            </select>
          </div>
        </div>
      </nav>
      <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
        <div
          class="row"
          v-for="(imdata,idx) in client_data.session.gdata"
          :key="idx"
        >
          <div
            v-if="client_data.session.timing.length > 0"
            class="row"
          >
            <span>Search refinement took {{ client_data.session.timing[idx].toFixed(2) }} seconds</span>
          </div>
          <div class="row">
            <m-image-gallery
              ref="galleries"
              v-if="imdata.length > 0"
              :initial_imdata="filter_boxes(imdata, current_category)"
              @imdata-save="data_update(idx, $event)"
              @selection="handle_selection_change(idx, $event)"
              :refmode="refmode"
              @copy-ref="copy_ref(idx, $event)"
            />
          </div>
          <div class="row space" />
        </div>
        <div
          class="row"
          v-if="client_data.session.gdata.length > 0"
        >
          <button
            @click="next()"
            class="btn btn-dark btn-block"
          >
            More...
          </button>
        </div>
      </main>
    </div> 
  </div>
</template>
<script >
import MImageGallery from './components/m-image-gallery.vue';
import _ from 'lodash';

export default {
    components : {'m-image-gallery':MImageGallery},
    props: {},
    data () { return { 
                client_data : { session : { params :{ index_spec : {d_name:'', i_name:'', m_name:''}}, gdata : [] }, 
                                indices : [] 
                              },
                current_category : null,
                current_index : null,
                session_path : null,
                selection: null, 
                text_query:null,
                refmode : false,
                global_idx_view : [],
              }
            },
    mounted (){
        fetch('/api/getstate', {cache: "reload"})
            .then(response => response.json())
            .then(this._update_client_data)
    },
    methods : {
        total_images() {
            return this.client_data.session.gdata.map((l) => l.length).reduce((a,b)=>a+b, 0)
        },
        load_session(session_path){
            fetch(`/api/session_info`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify({path:session_path})}
            )
            .then(response => response.json())
            .then(this._update_client_data)
        },
        total_accepted() {
          let accepted_per_list = (l)=> l.map((elt) => elt.marked_accepted ? 1 : 0).reduce((a,b)=>a+b, 0)
          return this.client_data.session.gdata.map(accepted_per_list).reduce((a,b)=>a+b, 0)
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
            return this.client_data.session.gdata.map(annot_per_list).reduce((a,b)=>a+b, 0)
        },
        filter_boxes(imdata, category){
          let out = []
          for (const ent of imdata){
            let outent = {...ent};
            if (ent.refboxes != null){
              outent.refboxes = ent.refboxes.filter((b) => b.category === category || category === '');
            }
            out.push(outent);
          }
          return out
        },
        get_nth_idcs(global_idx){
          var rem = global_idx;
          for (const [gdata_idx, gdata] of this.client_data.session.gdata.entries()){
            console.assert(rem >= 0);

            if (rem < gdata.length){
              return {gdata_idx:gdata_idx, local_idx:rem};
            } else{
              rem -= gdata.length;
            }
          }
          console.assert(false, global_idx)
        },
        get_global_idx(gdata_idx, local_idx){
          var acc = 0
          console.assert(gdata_idx >= 0)
          for (const [curr_gdata_idx, gdata] of this.client_data.session.gdata.entries()){
            if (curr_gdata_idx === gdata_idx){
              console.assert(local_idx < gdata.length);
              return acc + local_idx
            } else {
              acc += gdata.length;
            }
          }
          console.assert(false, 'should not reach this', gdata_idx, local_idx);
        },
        handle_selection_change(idx, evdata){
          let {local_idx, ev, ...rest} = evdata;
          // if can move in the right direction
          // will disable current modal and enable a different one from here
          let total = this.total_images();
          let delta = (ev === 'ArrowLeft') ? -1 : ((ev === 'ArrowRight') ? 1 : 0)
          let glidx = this.get_global_idx(idx, local_idx)
          let target = glidx + delta;
          if (target >= 0 && target < total){
            console.log('about to switch'); 
            let {gdata_idx:new_gdata_idx, local_idx:new_local_idx} = this.get_nth_idcs(target);
            this.$refs.galleries[idx].close_modal();
            this.$refs.galleries[new_gdata_idx].$data.selection = new_local_idx; // sel new
          } else{
            console.log('not legal to switch', idx, local_idx, target);
          }

        },
        data_update(gdata_idx, ev){
          console.log('data_update', gdata_idx, ev.idx, ev.imdata)
          this.client_data.session.gdata[gdata_idx][ev.idx] = ev.imdata
        },
        copy_ref(gdata_idx, panel_idx){
          let refboxes = this.client_data.session.gdata[gdata_idx][panel_idx].refboxes;
          let frefboxes = refboxes.filter((b) => b.category === this.current_category || this.current_category === '')
          let newboxes = frefboxes.length == 0 ? null : frefboxes
          this.client_data.session.gdata[gdata_idx][panel_idx].boxes = newboxes
        },
        _update_client_data(data, reset = false){
          console.log('current data', this.$data);
          console.log('update client data', data, reset);
          this.client_data = data;
          this.current_index = data.session.params.index_spec;
          this.selection = null;
        },
        reset(index){
          console.log('start reset...', index, {...this.$data});
          let reqdata = {index:index};
          // this.$data = this.data()
          
          fetch(`/api/reset`,   
              {method: 'POST', 
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(reqdata)
          })
          .then(response => response.json())
          .then(data => this._update_client_data(data, true))
        },
        text(text_query){
            fetch(`/api/text?key=${encodeURIComponent(text_query)}`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify({})}
            )
            .then(response => response.json())
            .then(this._update_client_data)
        },
        next(){
          console.log(' this' , this);
          let body = { client_data : this.$data.client_data };

            fetch(`/api/next`, {method:'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(body) // body data type must match "Content-Type" header
                            })
            .then(response => response.json())
            .then(this._update_client_data)
        },
        save(){
          let body = { client_data : this.$data.client_data};
          fetch(`/api/save`, {method:'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(body) // body data type must match "Content-Type" header
                            })
            .then(response => response.json())
            .then(p => console.log('save response', p))
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