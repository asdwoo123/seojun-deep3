const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  lintOnSave: false,
  pluginOptions: {
    electronBuilder: {
      nodeIntegration: true,
    }
  },
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.txt$/,
          use: 'raw-loader'
        }
      ]
    }
  }
})
