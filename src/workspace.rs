// GSL License:
//
// Copyright (C) 1996, 1997, 1998, 1999, 2000, 2001, 2007 Brian Gough
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or (at
// your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

use num::Float;

/// Workspace for adaptive integrators
pub struct IntegrationWorkSpace<T: Float> {
    pub size: usize,
    pub nrmax: usize,
    pub i: usize,
    pub maximum_level: usize,
    pub alist: Vec<T>,
    pub blist: Vec<T>,
    pub rlist: Vec<T>,
    pub elist: Vec<T>,
    pub order: Vec<usize>,
    pub level: Vec<usize>,
}

impl<T: Float> IntegrationWorkSpace<T> {
    pub fn new(limit: usize) -> IntegrationWorkSpace<T> {
        IntegrationWorkSpace {
            size: 0,
            nrmax: 0,
            i: 0,
            maximum_level: 0,
            alist: vec![T::zero(); limit],
            blist: vec![T::zero(); limit],
            rlist: vec![T::zero(); limit],
            elist: vec![T::zero(); limit],
            order: vec![0; limit],
            level: vec![0; limit],
        }
    }
    pub fn recieve(&self) -> (T, T, T, T) {
        let i = self.i;
        (self.alist[i], self.blist[i], self.rlist[i], self.elist[i])
    }
    pub fn append_interval(&mut self, a: T, b: T, area: T, error: T) {
        let inew = self.size;
        self.alist[inew] = a;
        self.blist[inew] = b;
        self.rlist[inew] = area;
        self.elist[inew] = error;
        self.order[inew] = inew;
        self.level[inew] = 0;
        self.size += 1;
    }
    pub fn update(&mut self, point1: (T, T, T, T), point2: (T, T, T, T)) {
        let (a1, b1, area1, error1) = point1;
        let (a2, b2, area2, error2) = point2;

        let imax = self.i;
        let inew = self.size;
        let newlevel = self.level[imax] + 1;

        if error2 > error1 {
            self.alist[imax] = a2;
            self.rlist[imax] = area2;
            self.elist[imax] = error2;
            self.level[imax] = newlevel;

            self.alist[inew] = a1;
            self.blist[inew] = b1;
            self.rlist[inew] = area1;
            self.elist[inew] = error1;
            self.level[inew] = newlevel;
        } else {
            self.blist[imax] = b1;
            self.rlist[imax] = area1;
            self.elist[imax] = error1;
            self.level[imax] = newlevel;

            self.alist[inew] = a2;
            self.blist[inew] = b2;
            self.rlist[inew] = area2;
            self.elist[inew] = error2;
            self.level[inew] = newlevel;
        }

        self.size += 1;
        if newlevel > self.maximum_level {
            self.maximum_level = newlevel;
        }
        self.sort();
    }
    pub fn large_interval(&self) -> bool {
        self.level[self.i] < self.maximum_level
    }
    pub fn increase_nrmax(&mut self) -> bool {
        let id = self.nrmax;
        let last = self.size - 1;
        let limit = self.alist.len();

        let jupbnd = if last > (1 + limit / 2) {
            limit + 1 - last
        } else {
            last
        };

        for _k in id..(jupbnd + 1) {
            let imax = self.order[self.nrmax];
            self.i = imax;
            if self.level[imax] < self.maximum_level {
                return true;
            }
            self.nrmax += 1;
        }
        false
    }
    pub fn reset_nrmax(&mut self) {
        self.nrmax = 0;
        self.i = self.order[0];
    }
    pub fn sum_results(&self) -> T {
        self.rlist
            .iter()
            .take(self.size)
            .fold(T::zero(), |acc, x| acc + *x)
    }
    pub fn sort(&mut self) {
        let last = self.size - 1;
        let limit = self.alist.len();

        let mut i_nrmax = self.nrmax;
        let mut i_maxerr = self.order[i_nrmax];

        // Check whether the list contains more than two error estimates
        if last < 2 {
            self.order[0] = 0;
            self.order[1] = 1;
            self.i = i_maxerr;
            return;
        }

        let errmax = self.elist[i_maxerr];

        // This part of the routine is only executed if, due to a difficult
        // integrand, subdivision increased the error estimate. In the normal
        // case the insert procedure should start after the nrmax-th largest
        // error estimate.
        while i_nrmax > 0 && errmax > self.elist[self.order[i_nrmax - 1]] {
            self.order[i_nrmax] = self.order[i_nrmax - 1];
            i_nrmax -= 1;
        }

        // Compute the number of elements in the list to be maintained in
        // descending order. This number depends on the number of subdivisions
        // still allowed.
        let top = if last < (limit / 2 + 2) {
            last
        } else {
            limit - last + 1
        };

        // Insert errmax by traversing the list top-down, starting comparison
        // from the element elist[order[i_nrmax+1]]
        let mut i = i_nrmax + 1;

        // The order of the tests in the following line is import to prevent
        // segmentation fault
        while i < top && errmax < self.elist[self.order[i]] {
            self.order[i - 1] = self.order[i];
            i += 1;
        }
        self.order[i - 1] = i_maxerr;

        // Insert errmin by traversing the list bottom-up
        let errmin = self.elist[last];
        let mut k = (top - 1) as i32;
        let ii = i as i32;

        while k > ii - 2 && errmin >= self.elist[self.order[k as usize]] {
            self.order[(k + 1) as usize] = self.order[k as usize];
            k -= 1;
        }
        let k = k as usize;

        self.order[k + 1] = last;

        // Set i_max and e_max
        i_maxerr = self.order[i_nrmax];
        self.i = i_maxerr;
        self.nrmax = i_nrmax;
    }
    pub fn sort_results(&mut self) {
        let nint = self.size;
        for i in 0..nint {
            let i1 = self.order[i];
            let mut e1 = self.elist[i1];
            let mut imax = i1;

            for j in (i + 1)..nint {
                let i2 = self.order[j];
                let e2 = self.elist[i2];
                if e2 >= e1 {
                    imax = i2;
                    e1 = e2;
                }
            }
            if imax != i1 {
                self.order[i] = self.order[imax];
                self.order[imax] = i1;
            }
        }
        self.i = self.order[0];
    }
}
