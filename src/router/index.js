import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import Train from '../components/Train.vue'
import Predict from '../components/Predict.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
    children: [
      {
        path: '', name: 'home.default', component: Train
      },
      {
        path: '/train', name: 'train', component: Train
      },
      {
        path: '/predict', name: 'predict', component: Predict
      }
    ]
  },
  {
    path: '/about',
    name: 'about',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../views/AboutView.vue')
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
