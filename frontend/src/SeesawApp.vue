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
        ref="text_input"
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
    <!-- <div v-show="show_config" 
      class="container"> 
      <m-config-vue-3
        ref="config"
        v-bind:client_data="client_data.default_params"
      />
    </div> -->
    <div class="container-fluid">
      <nav
        id="sidebarMenu"
        class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse"
      >
        <div class="position-sticky pt-3">
          <div class="row">
            <div class="col">
              <!-- <div class="row">
                <label>Current Database:</label>
              </div> -->
              <!-- <div class="row">
                <select
                  v-model="selected_index"
                  @change="reset(selected_index)"
                >
                  <option
                    v-for="(idxspec,idx) in [null, ...client_data.indices]"
                    :key="idx"
                    :value="idxspec"
                  >
                    {{ idxspec != null ? `${idxspec.d_name}:${idxspec.i_name}` : '' }}
                  </option>
                </select> 
              </div> 
            -->
            </div>
          </div>
          <div class="row" v-if="client_data.session != null">
            <span>Current database: {{selected_index.d_name}}</span>
          </div>
          <div class="row" v-if="client_data.session != null">
            <span>Total images: {{ total_images() }}</span>
          </div>
          <div class="row" v-if="client_data.session != null">
            <span>Total accepted: {{ total_accepted() }}</span>
          </div>
          <!-- <div class="row" v-if="client_data.session != null">
            <button 
              class="btn btn-dark btn-block" 
              @click="save()"
            > 
              Save 
            </button>
          </div> -->
          <!-- <div class="row" v-if="client_data.session != null">
            <button
              class="btn btn-dark btn-block"
              @click="reset(client_data.session.params.index_spec)"
            >
              Reset
            </button>
          </div> -->
          <div class="row">
            <button
              v-if="show_config"
              class="btn btn-dark btn-block" 
              @click="toggle_config()"> 
              Hide Config
            </button>
            <button
              v-else
              class="btn btn-dark btn-block" 
              @click="toggle_config()"> 
              Show Config
            </button>
          </div>
        </div>
      </nav>
      <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4" v-if="client_data.session != null">
        <div v-if="session_path !== null" class="row">
          <a :href="`${session_path}/stdout`">stdout</a>
          <a :href="`${session_path}/stderr`">stderr</a>
        </div>

        <div v-if="other_url !== null" class="row">
          <a :href="other_url">{{other_url}}</a>
        </div>

        <div
          class="row"
          v-for="(imdata,idx) in client_data.session.gdata"
          :key="idx"
        >
          <!-- <div
            v-if="client_data.session.timing.length > 0"
            class="row"
          >
            <span>Search refinement took {{ client_data.session.timing[idx].toFixed(2) }} seconds</span>
          </div> -->
          <div class="row">
            <m-image-gallery
              ref="galleries"
              v-if="imdata.length > 0"
              :initial_imdata="imdata"
              :imdata_keys="imdata.map((imdat) => get_vue_key(imdat.dbidx))"
              @selection="handle_selection_change({gdata_idx:idx, local_idx:$event})"
            />
          </div>
          <!-- <div class="row space" /> -->
        </div>
        <!-- <div
          class="row"
          v-if="client_data.session.gdata.length > 0"
        >
          <button
            @click="next(false)"
            class="btn btn-dark btn-block"
          >
            Load More Images
          </button>
        </div> -->
      </main>
    </div> 
    <m-modal
      v-if="selection != null"
      ref="modal"
      @modalKeyDown="handleModalKeyUp('down', $event)"
      @modalKeyUp="handleModalKeyUp('up', $event)"
    >      
      <div class="keyword-text">
        <span> Object We Are Looking For: {{this.client_data.session.query_string}} </span>
      </div> 
      <div
        v-if="annotator_text_pointer != null"
      >
        <div v-if="front_end_type !== 'plain'">
          <div v-if="front_end_type !== 'pytorch'">
            <button
              class="btn btn-danger"
              v-if="annotator_text_pointer.box.data.marked_accepted"
              @click="toggle_box_accepted()"
              onfocus="blur()"
            >
              Mark Negative
            </button>
            <button
              class="btn btn-danger"
              v-else
              @click="toggle_box_accepted()"
              onfocus="blur()"
            >
              Mark Accepted
            </button>
          </div>
          <Autocomplete 
              v-if="this.front_end_type === 'textual'"
              @input="changeInput"
              @onSelect="inputSelect"
              :results="autocomplete_items"
              :placeholder="annotator_text"
              :results-container-class="['custom-vue3-results-container']"
              />
        </div>
      </div>
      <div>
        <button
            class="btn btn-danger"
            onfocus="blur()"
          >
            See Examples (Q)
        </button>
        <div
        class="button-row" 
        v-if="front_end_type === 'pytorch' && allow_full_box">
          <button
            v-if="checkForFullBox()"
            class="btn btn-danger"
            @click="mark_image_accepted()"
            onfocus="blur()"
          >
            Select Full Box (W)
          </button>
          <button
            v-else
            class="btn btn-danger"
            @click="mark_image_accepted()"
            onfocus="blur()"
          >
            Create Full Box (W) 
          </button>
        </div>
          <button
            class="btn btn-warning highlight-button"
            v-if="front_end_type !== 'plain'"
            ref="highlight_btn"
            @click="() => {this.$refs.highlight_btn.blur()}"
          >
            Toggle Highlight (E)
          </button>
      </div>
      <div> 
        <button
            class="btn btn-danger"
            ref="left_button"
            :disabled="this.image_index === 1 || this.image_index === null"
            onfocus="blur()"
          >
            Previous (A)
        </button>
        <div 
        class="button-row"
        v-if="front_end_type === 'plain'">
          <button
            class="btn btn-danger"
            ref="accept_button"
            onfocus="blur()"
          >
            Toggle Accept (S)
          </button>
        </div>
        <div
        class="button-row" 
        v-else-if="front_end_type === 'textual'">
          <button
            class="btn btn-danger"
            @click="create_full_box()"
            onfocus="blur()"
          >
            Full Box
          </button>
        </div>
        <button
            class="btn btn-danger"
            v-if="front_end_type !== 'plain'"
            :disabled="annotator_text_pointer == null"
            onfocus="blur()"
          >
            Delete Box (S)
          </button>
        <button
            class="btn btn-danger"
            ref="right_button"
            onfocus="blur()"
            :disabled="loading_next"
          >
            Next (D)
        </button>
      </div>
      <div class="keyword-text">
        <span> Image {{this.image_index}} of {{this.total_images()}} </span>
      </div>
      <div class="row">
        <m-annotator
          ref="annotator"
          :initial_imdata="this.client_data.session.gdata[this.selection.gdata_idx][this.selection.local_idx]"
          :read_only="false"
          :front_end_type="this.front_end_type"
          @selection="handleAnnotatorSelectionChange($event)"
          :key="get_vue_key(this.client_data.session.gdata[this.selection.gdata_idx][this.selection.local_idx].dbidx)"
          :app_handle="this"
        />
      </div>

    </m-modal>
    <m-modal
      v-if="end_query == true"
      ref="notif-modal"
      @modalKeyDown="handleModalKeyUp('down', $event)"
      @modalKeyUp="handleModalKeyUp('up', $event)"
    > 
      <div>
      <div class="keyword-text">
        <span> {{this.notif_description}} </span>
      </div> 
      <m-example-image-gallery 
        v-bind:urls="example_urls"/>
      <button
            class="btn btn-danger"
            onfocus="blur()"
            @click="load_next_task()"
            :disabled="!next_task_ready"
          >
            OK
        </button>
        </div>
    </m-modal>
  </div>  
</template>
<script lang="ts">
import {callWithAsyncErrorHandling, defineComponent} from 'vue';
import {FrontendT, frontends} from './frontend_types'

import MImageGallery from './components/m-image-gallery.vue';
import MAnnotator from './components/m-annotator.vue';
import MModal from './components/m-modal.vue';

import MConfigVue3 from './components/m-config-vue3.vue';

import MExampleImageGallery from './components/m-example-image-gallery.vue'; 

import Autocomplete from 'vue3-autocomplete'
// Optional: Import default CSS
import 'vue3-autocomplete/dist/vue3-autocomplete.css'

import {image_accepted, getCookie, setCookie} from './util'

export default defineComponent({
    components : {'m-image-gallery':MImageGallery, 'm-modal':MModal, 'm-annotator':MAnnotator, MConfigVue3, Autocomplete, MExampleImageGallery},
    props: {},
    data () { return { 
                client_data : { session : null,
                              // WHEN NOT NULL, SESSION HAS THIS SHAPE  
                              // { params :{ index_spec : {d_name:'', i_name:'', m_name:''}}, 
                              //                     gdata : [] 
                              //  query_string
                              //                     },

                                indices : [],
                                default_params : {}
                              },
                selected_index : null,
                session_path : null,
                selection: null, 
                text_query:null,
                imdata_knum : {},
                keys : {},
                new_session_params : null,
                annotator_text : '',
                annotator_text_pointer : null,
                show_config : false,
                other_url : null,
                autocomplete_items: [],
                task_info : null,
                front_end_type : 'textual', // by default textual so it works when using plain url
                button_labels : {
                  "test" : {"add": "Add Button"},
                }, 
                image_index : null, 
                loading_next : false,
                allow_full_box : false, 
                end_query : false,  
                example_urls : null, 
                notif_description : '',
                next_task_ready : false, 
                task_started : false, 
                task_start_time : Date.now(),
              }
            },
    mounted (){
        window.VueApp = this;
        let params = new URLSearchParams(window.location.search)

        if (window.location.pathname === '/session_info'){
            let session_path = params.get('path')
            this.load_session(session_path)
        } else if (window.location.pathname === '/compare'){
            let session_path = params.get('path')
            this.other_url = `${window.location.origin}/session_info?path=${params.get('other')}`
            this.load_session(session_path)
        } else if (window.location.pathname === '/user_session'){
            // set front-end mode here based on user session params
            fetch('/api/user_session?' + params, {method: 'POST'})
            .then(response => response.json())
            .then(this._update_client_data)
        }  else if (window.location.pathname === '/session') {
          // get task list metadata for loop. cookie will be initialized on return if it doesn't exist
            fetch('/api/session?' + params, {method:'POST'})
            .then(response => response.json())
            .then(this._update_client_data)
        } else{
          fetch('/api/getstate', {cache: "reload"})
              .then(response => response.json())
              .then(this._update_client_data)
        }
        this.checkContainer(); 
    },
    methods : {
      finish_task(){
        //var value = this.image_index == 5; 
        var value = ((this.total_accepted() == 10) || (Math.floor((Date.now() - this.task_start_time)/1000) > 60 * 4)); 
        console.log("Accepted Images: ", this.total_accepted()); 
        console.log("Change time: ", Math.floor((Date.now() - this.task_start_time)/1000)); 
        return value; 
      }, 
      nextButtonClick(){
        if (this.finish_task()){ // End Task
          this.close_modal(); 
          this.get_end_description(); 
        }
        if (this.image_index < this.total_images()){
          this.moveRight()
        } else if (!this.loading_next){
          this.next(true)
        }
      },
      log(message : string, other_fields : Object) {
        this.client_data.session.action_log.push({logger:'client', time:Date.now()/1000, seen:this.total_images(), accepted:this.total_accepted(),
         message:message, other_fields : other_fields});
      },
      checkContainer () {
        let input = document.querySelector('.vue3-input');
        if(input !== null){ //if the container is visible on the page
          input.focus(); 
        } 
        setTimeout(this.checkContainer, 1000); //wait 50 ms, then try again
      },
      mark_image_accepted(){
          this.$refs.annotator.draw_full_frame_box(true); 
      }, 
      toggle_box_accepted(){
        if (!this.annotator_text_pointer.box.data.marked_accepted){
          this.annotator_text_pointer.box.data.marked_accepted = true; 
          this.annotator_text_pointer.box.strokeColor = 'green'; 
          this.annotator_text_pointer.description.fillColor = 'green'; 
        } else {
          this.annotator_text_pointer.box.data.marked_accepted = false; 
          this.annotator_text_pointer.box.strokeColor = 'red'; 
          this.annotator_text_pointer.description.fillColor = 'red'; 
        }
      }, 
      image_accepted(imdata){ // make it accessible from the <template>
          return image_accepted(imdata)
      },
      updateRecommendations() {
        this.autocomplete_items = []; 
        if (this.client_data.session) {
          for (var row of this.client_data.session.gdata){
            for (var item of row){
              if (item.boxes !== null){
                for (var box of item.boxes){
                  if (!this.autocomplete_items.includes(box.description) 
                      && box.description !== ""
                      && box.description !== null){
                    this.autocomplete_items.push(box.description); 
                  }
                }
              }
            }
          }
       }
      },
      set_frontend_type(mode){
        switch (mode) {
                case 'default':
                  this.front_end_type = 'plain';
                  break;
                case 'pytorch':
                case 'fine':
                  this.front_end_type = 'pytorch';
                  break
                default:
                  console.log(`unknown mode ${params.get('mode')}.`);
            }
      } ,
      changeInput(input){
        console.log("change Input" + input); 
        this.annotator_text = input; 
        this.autocomplete_items = this.autocomplete_items.filter((item) => {
          if (item.includes(input)){
            return item
          }
        })
      }, 
      get_session_id() {
        return getCookie('session_id');
      },
      set_session_id(session_id : string){
        if (session_id){ // don't set it to undefined or null, bc it will become
          console.log('setting session id', session_id);
          setCookie('session_id', session_id, 1);
        }
      },
      delete_session_id(){  // useful for undoing things
        setCookie('session_id', '', 0);
      },
      inputSelect(input){
        console.log("input Select: " + input); 
        this.annotator_text = input; 
        this.annotator_text_pointer.description.content = this.annotator_text
        this.annotator_text_pointer = null;
      }, 
        total_images() {
            return this.client_data.session.gdata.map((l) => l.length).reduce((a,b)=>a+b, 0); 
        },
        toggle_config() { 
          this.show_config = !this.show_config; 
        }, 
        load_session(session_path){
            this.session_path = session_path
            fetch(`/api/session_info`,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify({path:session_path})}
            )
            .then(response => response.json())
            .then(this._update_client_data)
        },
        get_end_description(){
          let index = this.client_data.worker_state.current_task_index; 
          this.task_started = false; 
          if (index != -1){
            this.log("task.end"); 
          }
          if (index == this.client_data.worker_state.task_list.length - 1){
            this.finish_session(); 
          } else {
            fetch('/api/task_description?code='+this.client_data.worker_state.task_list[index + 1].qkey,   
                  {method: 'GET'}
              )
              .then(response => response.json())
              .then(this._update_notify_module)
          }
        }, 
        finish_session(){
          let body = { client_data : this.$data.client_data };
          fetch(`/api/session_end`,   
                  {method: 'POST', 
                  headers: {'Content-Type': 'application/json'}, 
                  body: JSON.stringify(body)}
              )
              .then(response => response.json())
              .then(this._finish_session_data)
        }, 
        next_task(){
          let index = this.client_data.worker_state.current_task_index; 
          this.next_task_ready = false; 
          if (index == this.client_data.worker_state.task_list.length - 1){
            console.log("last session"); 
          } else {
            let body = { client_data : this.$data.client_data };
            fetch(`/api/next_task`,   
                  {method: 'POST', 
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify(body)}
              )
              .then(response => response.json())
              .then(this._update_client_data)
          }
        }, 
        load_next_task(){
          //this._update_client_data(this.next_task);
          this.end_query = false;
          if (this.task_started == false){ // if we haven't started a task, do the auto enter
            this.text(this.text_query); 
          } 
          this.task_started = true;   
        }, 
        total_accepted() {
          let accepted_per_list = (l)=> l.map((elt) => image_accepted(elt) ? 1 : 0).reduce((a,b)=>a+b, 0)
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
    handle_selection_change(new_selection){
      //console.log(this.idx, $event}); 
      if (this.$refs.annotator != undefined){
        let imdata = this.$refs.annotator.get_latest_imdata();
        this.data_update(imdata);
      }
      if (this.selection != new_selection && this.$refs.annotator != undefined){
        this.$refs.annotator.annotator_end(); 
      }
      this.log('selection.end', this.selection);
      this.selection = new_selection;
      this.log('selection.start', new_selection);
      if (this.selection !== null){
        this.image_index = this.get_global_idx(this.selection.gdata_idx, this.selection.local_idx) + 1; 
      } else {
        this.image_index = null; 
      }
    },
    toggle_plain_accepted(){
      if (this.checkForFullBox()){
        this.delete_full_box(); 
      } else {
        this.mark_image_accepted(); 
      }
    }, 
    toggle_mark_accepted(){
      if (this.annotator_text_pointer.box.data.marked_accepted){
          this.annotator_text_pointer.box.strokeColor = 'green'
      } else {
          this.annotator_text_pointer.box.strokeColor = 'yellow'
      }
    },
    handleAnnotatorSelectionChange(ev){
      console.log('annotator sel change', ev)
      if (this.annotator_text_pointer != null){ // save form state into paper box
          this.annotator_text_pointer.description.content = this.annotator_text
      }

      if (ev != null){ 
        console.log("predicted path"); 
        this.annotator_text_pointer = ev; 
        this.annotator_text = this.annotator_text_pointer.description.content;
        console.log(this.annotator_text_pointer); 
      } else {
        this.annotator_text_pointer = null;
      }
      this.updateRecommendations(); 
    },
    checkForFullBox(){
      if (this.$refs.annotator === null || this.$refs.annotator === undefined){
        return false; 
      }
      let retval =  this.$refs.annotator.full_box_present(); 
      console.log('check for full box returning', retval, this.$refs.annotator)
      return retval
    }, 
    moveLeft(){
      if (!(this.image_index === 1 || this.image_index === null)){
        let delta =  -1;
        this.handle_arrow(delta);
      }
      //var element = this.$refs.left_button
      //element.blur()
    }, 
    moveRight(){
      let delta =  1;
      this.handle_arrow(delta);
      //var element = this.$refs.right_button
      //element.blur()
    },
    sButtonClick(){
      if (this.front_end_type !== 'plain' && this.annotator_text_pointer !== null){
        this.delete_annotation(); 
      } else if (this.front_end_type === 'plain'){
        this.toggle_plain_accepted(); 
      }
    }, 
    toggleActivation(){
      if (this.front_end_type === 'pytorch'){
        this.$refs.highlight_btn.blur(); 
        this.$refs.annotator.activation_press();
      }
    }, 
    handleModalKeyUp(up_or_down, ev){
      if (up_or_down === 'up') {
        console.log('within modalKeyUp handler', ev)
        //if (this.annotator_text_pointer == null){ // ie if text is being entered ignore this
        if (this.annotator_text_pointer == null || this.front_end_type !== 'textual'){ // ie if text is being entered ignore this
          console.log("EV CODE"); 
          console.log(ev.code); 
          if (ev.code === 'KeyD'){
            this.nextButtonClick(); 
          } else if (ev.code === 'KeyA'){
            this.moveLeft(); 
          } else if (ev.code == 'KeyW'){
            // TODO: make it toggle accept the image
            if (this.front_end_type === 'pytorch' && this.allow_full_box){
              this.mark_image_accepted(); 
            } 
            //this.mark_image_accepted(); 
          }  else if (ev.code == 'KeyE'){
            this.toggleActivation(); 
          } else if (ev.code == 'KeyS'){
            this.sButtonClick(); 
          } else if (ev.code == 'KeyQ'){
            this.end_query = true; 
          } else if (ev.code == 'Space'){
            //this.next(); 
          }
        } else { // assume text
          if (ev.code == 'Escape'){
            this.handleAnnotatorSelectionChange(null) // save text
            this.close_modal();
          } else if (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight'){
            this.handleAnnotatorSelectionChange(null) // save text
            let delta = (ev.code === 'ArrowLeft') ? -1 : 1
            this.handle_arrow(delta);
          } else if (ev.code == 'Enter'){ // show the text in the
            this.handleAnnotatorSelectionChange(this.annotator_text_pointer) 
          }
        }
      }
    },
    delete_annotation(){
          this.$refs.annotator.delete_paper_obj(this.annotator_text_pointer);
          this.handleAnnotatorSelectionChange(null);
    },
    delete_full_box(){
      if(this.checkForFullBox()){
        this.$refs.annotator.draw_full_frame_box(true); 
        this.delete_annotation(); 
      }
    }, 
    create_full_box(){
      //TODO
      console.log("create full box ran");
      this.$refs.annotator.draw_full_frame_box(false); 
    }, 
    handle_arrow(delta){
      if (this.selection  != null){
          let {gdata_idx:idx, local_idx} = this.selection;
          // if can move in the right direction
          // will disable current modal and enable a different one from here
          let total = this.total_images();
          let glidx = this.get_global_idx(idx, local_idx)
          let target = glidx + delta;
          if (target >= 0 && target < total){
            console.log('about to switch'); 
            this.annotator_text_pointer = null; 
            let new_idx = this.get_nth_idcs(target);
            this.handle_selection_change(new_idx);
            // this.$refs.galleries[idx].close_modal();
            // this.$refs.galleries[new_gdata_idx].$data.selection = new_local_idx; // sel new
          } else{
            console.log('not legal to switch', idx, local_idx, target);
          }
      }
    },
    get_vue_key(dbidx){
      let vnum = 0;
        if (dbidx in this.imdata_knum){
          vnum = this.imdata_knum[dbidx]
        }
        return dbidx * 10000 + vnum; // needs to be unique per image as well
    },
    incr_vue_key(dbidx){
        if (! (dbidx in this.imdata_knum )){
          this.imdata_knum[dbidx] = 1;
        } else {
          this.imdata_knum[dbidx] += 1;
        }
    },
        data_update(imdata){
          console.log('data_update', imdata)
          this.client_data.session.gdata[this.selection.gdata_idx][this.selection.local_idx] = imdata;
          this.incr_vue_key(imdata.dbidx)
          this.updateRecommendations(); 
        },
        _update_next_task(data){
          console.log("Update Next Task Data: ", data); 
        },  
        _update_notify_module(data){
          console.log("Notify Module Data: ", data); 
          this.notif_description = data.description; 
          this.example_urls = data.urls; 
          this.end_query = true; 
          this.next_task(); 
        }, 
        _finish_session_data(data){
          this.notif_description = "Completed Survey. Survey Code: " + data.token; 
          this.end_query = true; 
          this.example_urls = null; 
          this.next_task_ready = false; 
        },
        _update_client_data(data, reset = false){
          console.log('current data', this.$data);
          console.log('update client data', data, reset);
          this.client_data = data;
          this.updateRecommendations(); 
          if (this.client_data.session != null){
            //this.$refs.config.updateClientData(data.session.params); 
            this.selected_index = this.client_data.session.params.index_spec;

            this.text_query = this.client_data.session.query_string
            this.set_frontend_type(this.client_data.session.params.other_params.mode);
            if ('qstr' in this.client_data.session.params.other_params &&
                  this.client_data.session.gdata.length == 0)
            {
              // set text query sent by server when it is sent by server and the session is new  
              this.text_query = this.client_data.session.params.other_params.qstr
            }

          } else {
            this.selected_index = null
          }

          if (this.client_data.worker_state.current_task_index == -1){
            this.get_end_description(); 
          }
          this.next_task_ready = true; 
          //this.handle_selection_change(null);
        },
        reset(index){
          let config = this.$refs.config.currentConfig();           
          let reqdata = {config: null};
          if (index != null){
            reqdata.config = {...config};
          }
          // this.$data = this.data()
          this.client_data.session = null; // clear current screen
          console.log('reset request', reqdata)
          fetch(`/api/reset`,   
              {method: 'POST', 
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(reqdata)
          })
          .then(response => response.json())
          .then(data => this._update_client_data(data, true))
        },
        text(text_query : string){
            // TODO: refactor this after user study
            let params = new URLSearchParams({key:text_query})
            fetch(`/api/text?` + params,   
                {method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify({})}
            )
            .then(response => response.json())
            .then(data => {
              console.log('text cb')
              this._update_client_data(data);
              if (this.selection === undefined || this.selection === null){
                    this.$refs.text_input.blur();
                    this.log("task.started"); 
                    this.task_start_time = Date.now(); 
                    this.handle_selection_change({gdata_idx:0, local_idx:0})
                  }
            })
            .catch(e => console.log(e))
        },
        next(move_right: boolean = false){
          let handle_ret = (new_data) => {
            this.log('next_req.end');
            console.log("NEW DATA: ", new_data); 
            this._update_client_data(new_data);
            this.loading_next = false;
            if (move_right){ 
                console.log('hanlding ret', move_right, 'this', this); 
                this.moveRight() 
            }   else {
              console.log('no move right');
            }
          }

          if (!this.loading_next){
            this.log('next_req.start');
            this.loading_next = true; 
            let body = { client_data : this.$data.client_data };

              fetch(`/api/next`, {method:'POST',
                              headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify(body) // body data type must match "Content-Type" header
                              })
              .then((response) => {
                return response.json(); 
                })
              .then(handle_ret)
              .catch((error) => {
                console.log("Error in next: ", error); 
                this.loading_next = false; 
              })
          } else { 
            console.log("PREVENTED NEXT DUE TO WAITING");
          }

        },
        close_modal(){
          this.handle_selection_change(null)
        }
    }
})
</script>
<style> 
.custom-vue3-results-container {
    position: relative;
    border: 1px solid black;
    z-index: 99;
    background: white;
  }
</style>
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

.button-row {
  display: inline; 
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

  .container {
    margin-left:20%; 
  }

  .drag-el {
    background-color: #fff;
    margin-bottom: 10px;
    padding: 5px;
  }
  
</style>