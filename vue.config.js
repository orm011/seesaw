module.exports = {
  lintOnSave: false,  
  devServer : {
    watchOptions  : { // avoid watching these large folders
      ignored: ['dist', 'node_modules', '.git', 'seesaw', 'notebooks', 'scripts'],
      poll: 500
    }
  },
}
