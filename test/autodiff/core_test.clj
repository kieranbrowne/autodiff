(ns autodiff.core-test
  (:refer-clojure :exclude  [* + - /])
  ;; (:import [autodiff.core Dual])
  (:require [clojure.test :refer :all]
            [autodiff.core :refer :all]))


(deftest basic
  (testing "Simple quadratic"
    (is (= 6 (:f' (#(* % %) (autodiff.core.Dual. 3 1)))))))
