(function(t){function e(e){for(var a,c,r=e[0],s=e[1],l=e[2],u=0,f=[];u<r.length;u++)c=r[u],Object.prototype.hasOwnProperty.call(i,c)&&i[c]&&f.push(i[c][0]),i[c]=0;for(a in s)Object.prototype.hasOwnProperty.call(s,a)&&(t[a]=s[a]);d&&d(e);while(f.length)f.shift()();return o.push.apply(o,l||[]),n()}function n(){for(var t,e=0;e<o.length;e++){for(var n=o[e],a=!0,r=1;r<n.length;r++){var s=n[r];0!==i[s]&&(a=!1)}a&&(o.splice(e--,1),t=c(c.s=n[0]))}return t}var a={},i={app:0},o=[];function c(e){if(a[e])return a[e].exports;var n=a[e]={i:e,l:!1,exports:{}};return t[e].call(n.exports,n,n.exports,c),n.l=!0,n.exports}c.m=t,c.c=a,c.d=function(t,e,n){c.o(t,e)||Object.defineProperty(t,e,{enumerable:!0,get:n})},c.r=function(t){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},c.t=function(t,e){if(1&e&&(t=c(t)),8&e)return t;if(4&e&&"object"===typeof t&&t&&t.__esModule)return t;var n=Object.create(null);if(c.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:t}),2&e&&"string"!=typeof t)for(var a in t)c.d(n,a,function(e){return t[e]}.bind(null,a));return n},c.n=function(t){var e=t&&t.__esModule?function(){return t["default"]}:function(){return t};return c.d(e,"a",e),e},c.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},c.p="/";var r=window["webpackJsonp"]=window["webpackJsonp"]||[],s=r.push.bind(r);r.push=e,r=r.slice();for(var l=0;l<r.length;l++)e(r[l]);var d=s;o.push([0,"chunk-vendors"]),n()})({0:function(t,e,n){t.exports=n("56d7")},1:function(t,e){},2:function(t,e){},"290f":function(t,e,n){},"56d7":function(t,e,n){"use strict";n.r(e);n("e260"),n("e6cf"),n("cca6"),n("a79d");var a=n("7a23"),i=n("2909"),o=(n("99af"),n("b680"),function(t){return Object(a["k"])("data-v-15626205"),t=t(),Object(a["j"])(),t}),c={class:"navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow"},r=o((function(){return Object(a["f"])("a",{class:"navbar-brand col-md-3 col-lg-2 me-0 px-3",href:"#"},"SeeSaw",-1)})),s=o((function(){return Object(a["f"])("button",{class:"navbar-toggler position-absolute d-md-none collapsed",type:"button","data-bs-toggle":"collapse","data-bs-target":"#sidebarMenu","aria-controls":"sidebarMenu","aria-expanded":"false","aria-label":"Toggle navigation"},[Object(a["f"])("span",{class:"navbar-toggler-icon"})],-1)})),l=o((function(){return Object(a["f"])("div",{class:"navbar-nav col-lg-1 px-3"},null,-1)})),d={class:"container-fluid"},u={id:"sidebarMenu",class:"col-md-3 col-lg-2 d-md-block bg-light sidebar collapse"},f={class:"position-sticky pt-3"},b={class:"row"},h={class:"col"},p=o((function(){return Object(a["f"])("div",{class:"row"},[Object(a["f"])("label",null,"Current Database:")],-1)})),m={class:"row"},_=["value"],v={class:"row"},g={class:"row"},j={class:"row"},O={class:"row"},y={class:"row"},x={class:"row"},w={key:0,class:"row"},k=o((function(){return Object(a["f"])("label",{for:"reference category"},"(DEBUG) pick ground truth category:",-1)})),S=["value"],P={class:"col-md-9 ms-sm-auto col-lg-10 px-md-4"},C={key:0,class:"row"},T={class:"row"},$=o((function(){return Object(a["f"])("div",{class:"row space"},null,-1)})),M={key:0,class:"row"};function R(t,e,n,o,R,I){var U=Object(a["n"])("m-image-gallery");return Object(a["i"])(),Object(a["e"])("div",null,[Object(a["f"])("header",c,[r,Object(a["t"])(Object(a["f"])("input",{class:"form-control form-control-dark w-auto",type:"text",placeholder:"Search","aria-label":"Search","onUpdate:modelValue":e[0]||(e[0]=function(t){return R.text_query=t}),onKeydown:e[1]||(e[1]=Object(a["u"])((function(t){return I.text(R.text_query)}),["enter"]))},null,544),[[a["r"],R.text_query]]),s,l]),Object(a["f"])("div",d,[Object(a["f"])("nav",u,[Object(a["f"])("div",f,[Object(a["f"])("div",b,[Object(a["f"])("div",h,[p,Object(a["f"])("div",m,[Object(a["t"])(Object(a["f"])("select",{"onUpdate:modelValue":e[2]||(e[2]=function(t){return R.current_index=t}),onChange:e[3]||(e[3]=function(t){return I.reset(R.current_index)})},[(Object(a["i"])(!0),Object(a["e"])(a["a"],null,Object(a["l"])(R.client_data.indices,(function(t,e){return Object(a["i"])(),Object(a["e"])("option",{key:e,value:t},Object(a["o"])(t),9,_)})),128))],544),[[a["q"],R.current_index]])])])]),Object(a["f"])("div",v,[Object(a["f"])("span",null,"Total images shown: "+Object(a["o"])(I.total_images()),1)]),Object(a["f"])("div",g,[Object(a["f"])("span",null,"Total images accepted: "+Object(a["o"])(I.total_accepted()),1)]),Object(a["f"])("div",j,[Object(a["f"])("button",{class:"btn btn-dark btn-block",onClick:e[4]||(e[4]=function(t){return I.save()})}," Save ")]),Object(a["f"])("div",O,[Object(a["f"])("button",{class:"btn btn-dark btn-block",onClick:e[5]||(e[5]=function(t){return I.reset(R.current_index)})}," Reset ")]),Object(a["f"])("div",y,[Object(a["t"])(Object(a["f"])("input",{"onUpdate:modelValue":e[6]||(e[6]=function(t){return R.session_path=t}),placeholder:"session path"},null,512),[[a["r"],R.session_path]])]),Object(a["f"])("div",x,[Object(a["f"])("button",{class:"btn btn-dark btn-block",onClick:e[7]||(e[7]=function(t){return I.load_session(R.session_path)})}," Load Session ")]),R.refmode?(Object(a["i"])(),Object(a["e"])("div",w,[k,Object(a["t"])(Object(a["f"])("select",{"onUpdate:modelValue":e[8]||(e[8]=function(t){return R.current_category=t})},[(Object(a["i"])(!0),Object(a["e"])(a["a"],null,Object(a["l"])([""].concat(Object(i["a"])(t.reference_categories)),(function(t,e){return Object(a["i"])(),Object(a["e"])("option",{key:e,value:t},Object(a["o"])(t),9,S)})),128))],512),[[a["q"],R.current_category]])])):Object(a["d"])("",!0)])]),Object(a["f"])("main",P,[(Object(a["i"])(!0),Object(a["e"])(a["a"],null,Object(a["l"])(R.client_data.session.gdata,(function(t,e){return Object(a["i"])(),Object(a["e"])("div",{class:"row",key:e},[R.client_data.session.timing.length>0?(Object(a["i"])(),Object(a["e"])("div",C,[Object(a["f"])("span",null,"Search refinement took "+Object(a["o"])(R.client_data.session.timing[e].toFixed(2))+" seconds",1)])):Object(a["d"])("",!0),Object(a["f"])("div",T,[t.length>0?(Object(a["i"])(),Object(a["c"])(U,{key:0,ref_for:!0,ref:"galleries",initial_imdata:I.filter_boxes(t,R.current_category),onImdataSave:function(t){return I.data_update(e,t)},refmode:R.refmode,onCopyRef:function(t){return I.copy_ref(e,t)}},null,8,["initial_imdata","onImdataSave","refmode","onCopyRef"])):Object(a["d"])("",!0)]),$])})),128)),R.client_data.session.gdata.length>0?(Object(a["i"])(),Object(a["e"])("div",M,[Object(a["f"])("button",{onClick:e[9]||(e[9]=function(t){return I.next()}),class:"btn btn-dark btn-block"}," More... ")])):Object(a["d"])("",!0)])])])}var I=n("5530"),U=n("b85c"),J=(n("d3b7"),n("d81d"),n("e9c4"),n("4de4"),{class:"row"}),q={class:"image-gallery"},N={class:"row"},V={class:"row"};function z(t,e,n,i,o,c){var r=Object(a["n"])("m-annotator"),s=Object(a["n"])("m-modal");return Object(a["i"])(),Object(a["e"])("div",null,[Object(a["f"])("div",J,[Object(a["f"])("div",q,[(Object(a["i"])(!0),Object(a["e"])(a["a"],null,Object(a["l"])(n.initial_imdata,(function(t,e){return Object(a["i"])(),Object(a["e"])("div",{key:c.imdata_key(e)},[Object(a["g"])(r,{class:Object(a["h"])(t.marked_accepted?"gallery-accepted":""),ref_for:!0,ref:"annotators",initial_imdata:t,read_only:!0,onCclick:function(t){return c.onclick(e)}},null,8,["class","initial_imdata","onCclick"])])})),128))])]),null!=t.selection?(Object(a["i"])(),Object(a["c"])(s,{key:0,ref:"modal",onClose:e[2]||(e[2]=function(t){return c.close_modal()}),tabindex:"0"},{default:Object(a["s"])((function(){return[Object(a["f"])("div",N,[(Object(a["i"])(),Object(a["c"])(r,{ref:"annotator",initial_imdata:n.initial_imdata[t.selection],key:c.imdata_key(t.selection),read_only:!1,tabindex:"1",onImdataSave:e[0]||(e[0]=function(e){return t.$emit("imdata-save",{idx:t.selection,imdata:e})})},null,8,["initial_imdata"]))]),Object(a["f"])("div",V,[n.refmode?(Object(a["i"])(),Object(a["e"])("button",{key:0,class:"btn btn-dark bton-block",onClick:e[1]||(e[1]=function(e){return t.$emit("copy-ref",t.selection)})}," Autofill ("+Object(a["o"])(n.initial_imdata[t.selection].refboxes.length)+" boxes) ",1)):Object(a["d"])("",!0)])]})),_:1},512)):Object(a["d"])("",!0)])}var K=["src"],B={class:"form-check annotator-check"},D=["disabled"],L={class:"form-check activation-check"};function W(t,e,n,i,o,c){var r=this;return Object(a["i"])(),Object(a["e"])("div",{class:"annotator_div row",ref:"container",onKeyup:e[9]||(e[9]=Object(a["u"])((function(e){return t.emit("esc")}),["esc"])),tabindex:"0"},[Object(a["f"])("img",{class:Object(a["h"])(n.read_only?"annotator_image_small":"annotator_image"),src:t.imdata.url,ref:"image",onLoad:e[0]||(e[0]=function(){return c.draw_initial_contents&&c.draw_initial_contents.apply(c,arguments)}),tabindex:"1",onKeyup:e[1]||(e[1]=Object(a["u"])((function(e){return t.emit("esc")}),["esc"]))},null,42,K),Object(a["f"])("canvas",{class:"annotator_canvas",ref:"canvas",onKeyup:e[2]||(e[2]=Object(a["u"])((function(e){return t.emit("esc")}),["esc"])),tabindex:"2",onClick:e[3]||(e[3]=function(){return c.canvas_click&&c.canvas_click.apply(c,arguments)}),onMouseover:e[4]||(e[4]=function(t){return c.hover(!0)}),onMouseleave:e[5]||(e[5]=function(t){return c.hover(!1)})},null,544),Object(a["f"])("div",B,[Object(a["t"])(Object(a["f"])("input",{class:"form-check-input",type:"checkbox",disabled:n.read_only,"onUpdate:modelValue":e[6]||(e[6]=function(t){return r.imdata.marked_accepted=t})},null,8,D),[[a["p"],this.imdata.marked_accepted]])]),Object(a["f"])("div",L,[Object(a["t"])(Object(a["f"])("input",{class:"form-check-input",type:"checkbox",id:"activation-checkbox","onUpdate:modelValue":e[7]||(e[7]=function(t){return r.show_activation=t}),onChange:e[8]||(e[8]=function(t){return r.activation_press()})},null,544),[[a["p"],this.show_activation]])])],544)}n("159b");var A=n("05b7"),E=n.n(A),G={name:"MAnnotator",props:["initial_imdata","read_only"],emits:["imdata-save"],data:function(){return{height_ratio:null,width_ratio:null,paper:null,imdata:this.initial_imdata,show_activation:!1,activation_paths:[]}},created:function(){console.log("created annotator")},mounted:function(){this.paper=new E.a.PaperScope,new E.a.Tool,console.log("mounted annotator"),this.paper2=new E.a.PaperScope,new E.a.Tool},methods:{activation_press:function(){this.show_activation?this.draw_activation():this.clear_activation()},draw_activation:function(){var t=this.$refs.image,e=this.$refs.container,n=t.height,a=t.width;e.style.setProperty("width",a+"px"),e.style.setProperty("height",n+"px"),t.style.setProperty("display","block");var i=this.paper2,o=this.$refs.canvas;o.height=n,o.width=a,i.setup(o),i.view.draw();for(var c=this.imdata.activation,r=(c=[[.5,.2,0],[.2,.1,0],[.1,0,0]],this.$refs.image.width/c[0].length),s=this.$refs.image.height/c.length,l=0;l<c[0].length;l++)for(var d=0;d<c.length;d++){var u=["Rectangle",r*l,s*d,r,s],f=i.Rectangle.deserialize(u),b=new i.Path.Rectangle(f);b.fillColor="red",b.strokeWidth=0,b.opacity=c[d][l],this.activation_paths.push(b)}i.view.draw(),i.view.update()},clear_activation:function(){while(0!==this.activation_paths.length){var t=this.activation_paths.pop();t.remove()}},rescale_box:function(t,e,n){var a=t.x1,i=t.x2,o=t.y1,c=t.y2;return{x1:a*n,x2:i*n,y1:o*e,y2:c*e}},save:function(){var t=this,e=this.paper,n=e.project.getItems({className:"Path"}).map((function(t){var e=t.bounds;return{x1:e.left,x2:e.right,y1:e.top,y2:e.bottom}})).map((function(e){return t.rescale_box(e,t.height_ratio,t.width_ratio)}));console.log("saving boxes"),0==n.length?(this.imdata.boxes=null,console.log("length 0 reverts to null right now")):this.imdata.boxes=n,this.$emit("imdata-save",this.imdata)},load_current_box_data:function(){var t=this.paper;if(null!=this.initial_imdata.boxes){console.log("drawing boxes",this.initial_imdata.boxes);var e,n=Object(U["a"])(this.initial_imdata.boxes);try{for(n.s();!(e=n.n()).done;){var a=e.value,i=this.rescale_box(a,this.height_ratio,this.width_ratio),o=["Rectangle",i.x1,i.y1,i.x2-i.x1,i.y2-i.y1],c=t.Rectangle.deserialize(o),r=new t.Path.Rectangle(c);r.strokeColor="green",r.strokeWidth=2,r.data.state=null,r.selected=!1,console.log("drew rect ",r)}}catch(s){n.e(s)}finally{n.f()}}},hover:function(t){this.read_only&&(this.$refs.image.style.opacity=t?.5:1)},toggle_accept:function(){this.imdata.marked_accepted=!this.imdata.marked_accepted},canvas_click:function(t){console.log("canvas click!",t),this.$emit("cclick",t)},draw_initial_contents:function(){console.log("(draw)setting up",this);var t=this.$refs.image,e=this.$refs.container,n=t.height,a=t.width;if(e.style.setProperty("width",a+"px"),e.style.setProperty("height",n+"px"),t.style.setProperty("display","block"),!this.read_only||null!==this.initial_imdata.boxes&&0!==this.initial_imdata.boxes.length){var i=this.paper,o=this.$refs.canvas;console.log("drawing canvas",t.height,t.width,t),o.height=n,o.width=a,i.setup(o),this.height_ratio=n/t.naturalHeight,this.width_ratio=a/t.naturalWidth,i.view.draw(),this.load_current_box_data(),this.read_only||this.setup_box_drawing_tool(i),i.view.draw(),i.view.update()}},setup_box_drawing_tool:function(t){var e=t.tool,n=function(e,n){var a=new t.Path.Rectangle(e,n);return a.strokeColor="green",a.strokeWidth=2,a.data.state=null,a.selected=!1,a};e.onMouseDown=function(e){var a={segments:!0,stroke:!0,fill:!0,class:t.Path,tolerance:10},i=t.project.hitTest(e.point,a),o=t.project.getSelectedItems();o.map((function(t){return t.selected=!1}));var c=null;if(!(null==i&&o.length>0))if(c=null==i?n(e.point.subtract(new t.Size(1,1)),e.point):i.item,c.selected=!0,null==i||"stroke"!==i.type){if(null==i||"segment"===i.type){c.data.state="resizing";for(var r=c.bounds,s=[["getTopLeft","getTopRight"],["getBottomLeft","getBottomRight"]],l=0;l<2;l++)for(var d=0;d<2;d++){var u=s[l][d],f=r[u]();if(f.isClose(e.point,a.tolerance)){var b=s[(l+1)%2][(d+1)%2];c.data.from=r[b](),c.data.to=r[u]()}}}}else c.data.state="moving"},e.onMouseUp=function(){},e.onKeyUp=function(e){var n=t.project.getSelectedItems();0!==n.length&&"d"===e.key&&n.forEach((function(t){return t.remove()}))},e.onMouseDrag=function(e){var n=t.project.getSelectedItems(),a=null;if(1===n.length)if(a=n[0],"moving"===a.data.state)a.position=a.position.add(e.point).subtract(e.lastPoint);else if("resizing"===a.data.state){var i=new t.Rectangle(a.data.from,e.point);0!==i.width&&0!==i.height&&(a.bounds=i)}}}}},F=(n("e304"),n("6b0d")),H=n.n(F);const Q=H()(G,[["render",W],["__scopeId","data-v-57f970a4"]]);var X=Q,Y={class:"my-modal-content",tabindex:"1"};function Z(t,e,n,i,o,c){return Object(a["i"])(),Object(a["e"])("div",{class:Object(a["h"])("my-modal ".concat(t.active?"my-modal-active":"")),tabindex:"0"},[Object(a["f"])("span",{class:"close",onClick:e[0]||(e[0]=function(){return c.close&&c.close.apply(c,arguments)})},"×"),Object(a["f"])("div",Y,[Object(a["m"])(t.$slots,"default",{},void 0,!0)])],2)}var tt={name:"MModal",data:function(){return{active:!0}},mounted:function(){},methods:{close:function(){this.$emit("close")},show:function(){}}};n("64d9");const et=H()(tt,[["render",Z],["__scopeId","data-v-20896a5e"]]);var nt=et,at={name:"MImageGallery",components:{"m-annotator":X,"m-modal":nt},props:{initial_imdata:{type:Array,default:function(){return[]}},refmode:Boolean},emits:["imdata-save","copy-ref"],data:function(){return{selection:null}},created:function(){},mounted:function(){},methods:{get_class:function(t){var e=this.initial_imdata[t];return null==e.boxes?"unknown":e.boxes.length>0?"accepted":0===e.boxes.length?"rejected":void 0},imdata_key:function(t){var e=this.initial_imdata[t],n=null==e.boxes?0:e.boxes.length+1,a=e.marked_accepted?1:0;return 1e3*t+100*a+n},close_modal:function(){this.$refs.annotator.save(),this.selection=null},onclick:function(t){this.selection=t}}};n("9bb4");const it=H()(at,[["render",z],["__scopeId","data-v-96ea8858"]]);var ot=it,ct={components:{"m-image-gallery":ot},props:{},data:function(){return{client_data:{session:{params:{index_spec:{d_name:"",i_name:"",m_name:""}},gdata:[]},indices:[]},current_category:null,current_index:null,session_path:null,selection:null,text_query:null,refmode:!1}},mounted:function(){fetch("/api/getstate",{cache:"reload"}).then((function(t){return t.json()})).then(this._update_client_data)},methods:{total_images:function(){return this.client_data.session.gdata.map((function(t){return t.length})).reduce((function(t,e){return t+e}),0)},load_session:function(t){fetch("/api/session_info",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:t})}).then((function(t){return t.json()})).then(this._update_client_data)},total_accepted:function(){var t=function(t){return t.map((function(t){return t.marked_accepted?1:0})).reduce((function(t,e){return t+e}),0)};return this.client_data.session.gdata.map(t).reduce((function(t,e){return t+e}),0)},total_annotations:function(){var t=function(t){var e,n=0,a=Object(U["a"])(t);try{for(a.s();!(e=a.n()).done;){var i=e.value;null!=i.boxes&&(n+=i.boxes.length)}}catch(o){a.e(o)}finally{a.f()}return n};return this.client_data.session.gdata.map(t).reduce((function(t,e){return t+e}),0)},filter_boxes:function(t,e){var n,a=[],i=Object(U["a"])(t);try{for(i.s();!(n=i.n()).done;){var o=n.value,c=Object(I["a"])({},o);null!=o.refboxes&&(c.refboxes=o.refboxes.filter((function(t){return t.category===e||""===e}))),a.push(c)}}catch(r){i.e(r)}finally{i.f()}return a},data_update:function(t,e){console.log("data_update",t,e.idx,e.imdata),this.client_data.session.gdata[t][e.idx]=e.imdata},copy_ref:function(t,e){var n=this,a=this.client_data.session.gdata[t][e].refboxes,i=a.filter((function(t){return t.category===n.current_category||""===n.current_category})),o=0==i.length?null:i;this.client_data.session.gdata[t][e].boxes=o},_update_client_data:function(t){var e=arguments.length>1&&void 0!==arguments[1]&&arguments[1];console.log("current data",this.$data),console.log("update client data",t,e),this.client_data=t,this.current_index=t.session.params.index_spec,this.selection=null},reset:function(t){var e=this;console.log("start reset...",t,Object(I["a"])({},this.$data));var n={index:t};fetch("/api/reset",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(n)}).then((function(t){return t.json()})).then((function(t){return e._update_client_data(t,!0)}))},text:function(t){fetch("/api/text?key=".concat(encodeURIComponent(t)),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})}).then((function(t){return t.json()})).then(this._update_client_data)},next:function(){console.log(" this",this);var t={client_data:this.$data.client_data};fetch("/api/next",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(t)}).then((function(t){return t.json()})).then(this._update_client_data)},save:function(){var t={client_data:this.$data.client_data};fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(t)}).then((function(t){return t.json()})).then((function(t){return console.log("save response",t)}))}}};n("875a");const rt=H()(ct,[["render",R],["__scopeId","data-v-15626205"]]);var st=rt,lt=(n("ab8b"),Object(a["b"])(st));lt.mount("#app")},"64d9":function(t,e,n){"use strict";n("e19d")},"875a":function(t,e,n){"use strict";n("290f")},9024:function(t,e,n){},"9bb4":function(t,e,n){"use strict";n("9024")},d035:function(t,e,n){},e19d:function(t,e,n){},e304:function(t,e,n){"use strict";n("d035")}});
//# sourceMappingURL=app.ce701f80.js.map