import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig(({command, mode}) => {
  console.log(command, mode);
  let config =  {
    server: {
      watch: {
        usePolling: true,
      }
    },
    plugins: [vue()]
  };

  return config

  // if (command === 'serve'){
  //   return config
  // } else {
  //   return config
  // }
}
)
