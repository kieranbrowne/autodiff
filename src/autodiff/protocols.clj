(ns autodiff.protocols
  (:refer-clojure :exclude [identity min max]))

(defprotocol AutoDiff
  (constant [u] "Create a constant of value u")
  (identity [u] "The function which returns itself")
  (add [u v] "Add two values")
  (sub [u v] "Subtract two values")
  (mul [u v] "Multiply two values")
  (matmul [u v] "Multiply two tensors")
  (negate [u] "Invert sign of value; ie -2 -> 2; 9.3 -> - 9.3")
  (abs [u])
  (signum [u])
  (div [u v] "Divide one value by another")
  (recip [u])
  (pi [type-like] "Return constant value of pi")
  (one [typed-thing] "Return constant value equivalent to 1")
  (two [typed-thing] "Return constant value equivalent to 1")
  (zero [typed-thing] "Return constant value equivalent to 1")
  (val-like [typed-thing v])
  (exp [u])
  (sqrt [u])
  (shape [u])
  (sigmoid [u])
  (log [u])
  (sin [u])
  (sum [u])
  (cos [u])
  (tan [u])
  (asin [u])
  (acos [u])
  (atan [u])
  (sinh [u])
  (cosh [u])
  (tanh [u])
  (asinh [u])
  (transpose [u])
  (acosh [u])
  (atanh [u])
  (pow [u v] "Raise one value to the power of another")
  (logbase [u v])
  )


(defrecord Dual
    [f f'])

(defn dual? [x]
  (= (type x) autodiff.protocols.Dual))

(defn coerce
  "Makes value a Dual if not already"
  ([x v]
   (if (dual? x) x
     (->Dual x (val-like x v))))
  ([x] (coerce x 0)))

(defmacro destruct-unary
  "Simpler let bindings for autodiff of Dual record type"
  [content]
  `(let ~[{'u :f 'u' :f' :or {'u 'u 'u' 0}} 'u]
     ~content))

(defmacro destruct-binary
  "Simpler let bindings for autodiff of Dual record type"
  [content]
  `(let ~[{'u :f 'u' :f'} '(coerce u)
          {'v :f 'v' :f'} '(coerce v)]
     ~content))


(extend-type Dual
  AutoDiff
  (constant [u]
    (destruct-unary
     (Dual. (constant u) (constant 0))))
  (identity [u]
    (destruct-unary
     (Dual. (identity u) (identity u'))))
  (add [u v]
    (destruct-binary
      (Dual. (add u v) (add u' v'))))
  (sub [u v]
    (destruct-binary
      (Dual. (sub u v) (sub u' v'))))
  (mul [u v]
    (destruct-binary
      (Dual. (mul u v) (add (mul u' v) (mul u v')))))
  (matmul [u v]
    (destruct-binary
     (Dual. (matmul u v)
            ;; (if (shape u) u')
            (cond ;(= (shape u) (shape v)) "fuck"
              (= (one u) u') ; if with respect to u
              (add (mul u (transpose (sum v')))
                   (mul (transpose (sum v)) u'))

              (= (one v) v') ; if with respect to v
              (transpose
               (add (mul (transpose v) (sum (transpose u')))
                    (mul (sum (transpose u)) (transpose v'))))

              true (throw (ex-info "Derivative must be with respect to something" {:u u :v v}))

                  )
            ;; [(add
            ;;   (mul u (transpose (sum v')))
            ;;   (mul (transpose (sum v)) u'))
            ;;  (add
            ;;   (mul u (transpose (sum v')))
            ;;   (mul (transpose (sum v)) u'))
            ;;  ]
             )))
            ;; (add (matmul u' v) (matmul u v')))))
  (div [u v]
    (destruct-binary
     (Dual. (div u v) (div (sub (mul u' v) (mul u v')) (mul v v)))))
  (log [u]
    (destruct-unary
     (Dual. (log u) (div u' u))))
  (pow [u v]
    (destruct-binary
     (Dual. (pow u v)
            (mul (pow u v)
                 (add (mul v' (log u))
                      (div (mul v u') u)
                      )))))
  (exp [u]
     (destruct-unary
      (Dual. (exp u) (mul u' (exp u)))))
  (sqrt [u]
    (destruct-unary
     (Dual. (sqrt u) (div u' (mul (sqrt u) (two u))))))
  (sin [u]
    (destruct-unary
     (Dual. (sin u) (mul u' (cos u)))))
  (sum [u]
    (destruct-unary
     (Dual. (sum u) (sum u'))))
  (cos [u]
    (destruct-unary
     (Dual. (cos u) (negate (mul u' (sin u))))))
  (tanh [u]
    (destruct-unary
     (Dual. (tanh u) (mul u' (sub (one u) (pow (tanh u) (two u)))))))
  (transpose [u]
    (destruct-unary
     (Dual. (transpose u) (transpose u'))))
  (sigmoid [u]
    (destruct-unary
     (Dual. (sigmoid u) (mul (sigmoid u) (sub u' (sigmoid u))))))
  (pi [type-like] (Dual. (pi type-like) (zero type-like)))
  (zero [type-like] (Dual. (one type-like) (zero type-like)))
  (one [type-like] (Dual. (one type-like) (zero type-like)))
  (two [type-like] (Dual. (two type-like) (zero type-like)))
  (val-like [typed-thing v] v)
  )

(defn wrt
  "'With Respect To'; sets the variable to count value"
  ([x nth-deriv] (coerce x nth-deriv))
  ([x] (coerce x 1)))

(defn d
  "Find the derivative of a function.
  All values are assumed to be constant.
  Use the wrt function to make the arg add to resulting derivative.
  "
  [f & args]
  (:f' (apply f (map #(coerce % 0) args))))
