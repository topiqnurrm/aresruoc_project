package com.littlelemon.menu

import org.junit.Test

import org.junit.Assert.*

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class FilterHelperTest {
    @Test
    fun filterProducts_filterTypeDessert_croissantReturned() {
        // arrange
        val sampleProductsList = mutableListOf(
            ProductItem(title = "Black tea", price = 3.00, category = "Drinks", R.drawable.black_tea),
            ProductItem(title = "Croissant", price = 7.00, category = "Dessert", R.drawable.croissant),
            ProductItem(title = "Bouillabaisse", price = 20.00, category = "Food", R.drawable.bouillabaisse)
        )

        // action
        val result = FilterHelper().filterProducts(FilterType.Dessert, sampleProductsList)

        // assert
        assertEquals(result.count(), 1)
        assertEquals(result.first().title, "Croissant")
    }
}