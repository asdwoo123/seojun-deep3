import { createApp } from 'vue'
import { VueDraggableResizable } from 'vue-draggable-resizable-vue3'
import Antd from 'ant-design-vue'
import App from './App.vue'
import store from './store'
import router from './router'

import 'ant-design-vue/dist/reset.css'
import '@/assets/common.scss'

createApp(App).use(router).use(store).use(Antd)
.use(VueDraggableResizable).mount('#app')
