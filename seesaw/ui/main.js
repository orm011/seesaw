const options = {
    moduleCache: {
      vue: Vue
    },
    async getFile(url) {
      const res = await fetch(url, {cache: "reload"});
      if ( !res.ok )
        throw Object.assign(new Error(res.statusText + ' ' + url), { res });
      return await res.text();
    },
    addStyle(textContent) {
      const style = Object.assign(document.createElement('style'), { textContent });
      const ref = document.head.getElementsByTagName('style')[0] || null;
      document.head.insertBefore(style, ref);
    },
  }

const { loadModule } = window['vue2-sfc-loader'];
console.log('hi');
loadModule('./app.vue', options)
.then(component => new Vue({components:{'app':component},
                            template : '<app/>' 
                            }).$mount('#app'));