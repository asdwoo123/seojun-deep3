import { createApp } from 'vue'
// import { VueDraggableResizable } from 'vue-draggable-resizable-vue3'
import Vue3DraggableResizable from 'vue3-draggable-resizable'
import Antd from 'ant-design-vue'
import App from './App.vue'
import store from './store'
import router from './router'

import 'ant-design-vue/dist/reset.css'
import 'vue3-draggable-resizable/dist/Vue3DraggableResizable.css'
import '@/assets/common.scss'

createApp(App).use(router).use(store).use(Antd)
.use(Vue3DraggableResizable).mount('#app')
