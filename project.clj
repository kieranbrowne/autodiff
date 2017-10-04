(defproject org.clojars.kieran/autodiff "0.1.1-SNAPSHOT"
  :description "Generic auto differentiation library for clojure"
  :url "http://kieranbrowne.com/"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]]
  :profiles {:test {:dependencies [[net.mikera/core.matrix "0.61.0"]]}}
  )
