import { StyleSheet, Text } from 'react-native'
import useFontSize from '../hooks/useFontSize'

export const BaseText = ({children}) =>{
  const fontStyle = useFontSize(16)
  return (
  <Text style={[styles.baseText, fontStyle]}>{children}</Text>
)}

const styles = StyleSheet.create({
  baseText: {
    marginInline: 'auto',
    textAlign: 'center',
  }
})