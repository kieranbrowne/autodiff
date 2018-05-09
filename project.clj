(defproject kieranbrowne/autodiff "0.1.5-SNAPSHOT"
  :description "Generic auto differentiation library for clojure"
  :url "https://github.com/kieranbrowne/autodiff"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.61.0"]]

  :profiles
  {:test
   {:dependencies [[net.mikera/core.matrix "0.61.0"]
                   [clojure-tensorflow "0.2.4"]]}
   :dev
   {:dependencies [[net.mikera/core.matrix "0.61.0"]
                   [clojure-tensorflow "0.2.4"]]}

   })
